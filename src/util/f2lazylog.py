#!/usr/bin/env python3
"""
f2lazylog.py

!!!!!!!!!!NOT COMPLETE. BREAKS FILES, METHODS SIGNATURES ETC.!!!!!!!!!!!

Преобразует f-строки внутри логирующих вызовов в ленивый формат:
logger.debug(f"...{x:.2f}...") -> logger.debug("...%.2f...", x)

Usage:
    python f2lazylog.py <path>          # file or directory (recursively)
    python f2lazylog.py --inplace <path>  # заменить файлы на месте (делает резервную копию .bak)

Ограничения:
- Обрабатывает JoinedStr (f-strings) и простые бинарные конкатенации (+) из Str/JoinedStr.
- Не трогает сложные выражения сборки строки (format(), f-strings в вложенных вызовах и т.п.).
- Всегда полезно сделать тестовый прогон и ревью diff'ов перед коммитом.
"""
import ast
import astor
import argparse
import os
import re
from typing import List, Tuple, Optional

LOG_METHODS = {"debug", "info", "warning", "warn", "error", "critical", "exception", "log"}

# Регекс для распознавания простых формат-спеков типа ".4f", "0.2f", ".0f"
_FMT_RE = re.compile(r'^(?P<dot>\.?)(?P<num>\d*)(?P<spec>[df])$')

class FstringToLazyLoggingTransformer(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        # Рекурсивно обработать аргументы/функцию
        self.generic_visit(node)

        # Определить имя метода, если это вызов вида logger.debug(...) или module.logger.debug(...)
        func = node.func
        method_name = None
        if isinstance(func, ast.Attribute):
            attr = func.attr
            if isinstance(attr, str) and attr in LOG_METHODS:
                method_name = attr
        elif isinstance(func, ast.Name):
            # прямой вызов debug(...) - редко, но игнорируем
            if func.id in LOG_METHODS:
                method_name = func.id

        if method_name is None:
            return node

        # Если нет позиционных аргументов, ничего делать не нужно
        if not node.args:
            return node

        first_arg = node.args[0]

        # попробуем собрать format_string и список выражений из first_arg (если это f-string / joinedstr / concat)
        res = self._extract_format_and_values(first_arg)
        if res is None:
            return node

        fmt_string, value_exprs = res

        # Создаём новый msg (строку) и добавляем value_exprs как последующие позиционные аргументы
        new_msg_node = ast.Constant(fmt_string)

        # сохраним оставшиеся позиционные аргументы (если были) — они идут после нашей группы аргументов
        remaining_pos_args = node.args[1:]  # остальные, если были

        # Новые args: [new_msg_node] + value_exprs + remaining_pos_args
        new_args = [new_msg_node] + value_exprs + remaining_pos_args

        new_call = ast.Call(func=node.func, args=new_args, keywords=node.keywords)
        return ast.copy_location(new_call, node)

    def _extract_format_and_values(self, node: ast.AST) -> Optional[Tuple[str, List[ast.AST]]]:
        """
        Попытаться преобразовать node (JoinedStr или BinOp конкатенация) в форматную строку + список выражений.
        Возвращает (fmt_string, [expr_nodes]) или None, если не получилось.
        """
        parts = []  # список кусочков литералов и placeholder'ов
        values: List[ast.AST] = []

        def append_literal(s: str):
            if s:
                parts.append(s)

        def handle_joinedstr(js: ast.JoinedStr) -> bool:
            for val in js.values:
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    append_literal(val.value)
                elif isinstance(val, ast.FormattedValue):
                    # получить формат-спек, если есть
                    spec = None
                    if val.format_spec is not None:
                        # format_spec может быть Constant(str) или JoinedStr
                        fs = None
                        if isinstance(val.format_spec, ast.Constant) and isinstance(val.format_spec.value, str):
                            fs = val.format_spec.value
                        elif isinstance(val.format_spec, ast.JoinedStr):
                            # попытаемся собрать простой spec
                            collected = []
                            for e in val.format_spec.values:
                                if isinstance(e, ast.Constant) and isinstance(e.value, str):
                                    collected.append(e.value)
                                else:
                                    return False
                            fs = "".join(collected)
                        else:
                            return False
                        spec = fs
                    ph = self._format_spec_to_placeholder(spec)
                    parts.append(ph)
                    values.append(val.value)
                else:
                    # неожиданный узел внутри JoinedStr
                    return False
            return True

        def handle_binop_concat(bop: ast.BinOp) -> bool:
            # разрешаем только цепочку + со строками / joinedstr
            # распакуем рекурсивно
            def rec(n):
                if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
                    return rec(n.left) and rec(n.right)
                elif isinstance(n, ast.Constant) and isinstance(n.value, str):
                    append_literal(n.value)
                    return True
                elif isinstance(n, ast.JoinedStr):
                    return handle_joinedstr(n)
                else:
                    return False
            return rec(bop)

        if isinstance(node, ast.JoinedStr):
            ok = handle_joinedstr(node)
            if not ok:
                return None
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # обычная строка — ничего делать не нужно
            return None
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            ok = handle_binop_concat(node)
            if not ok:
                return None
        else:
            return None

        # Собираем итоговую строку — экранируем '%'
        fmt = "".join(parts).replace("%", "%%")
        return fmt, values

    def _format_spec_to_placeholder(self, spec: Optional[str]) -> str:
        """
        Преобразуем формат-спек из f-string'а (без двоеточия) в printf-спек.
        Примеры:
          ".4f" -> "%.4f"
          "0.2f" -> "%0.2f"
          None -> "%s"
        По умолчанию: "%s"
        """
        if not spec:
            return "%s"
        # убрать возможные пробелы
        s = spec.strip()
        m = _FMT_RE.match(s)
        if m:
            # s/t: поддерживаем d и f
            return "%" + (m.group("num") or "") + ("." + m.group("num") if (m.group("dot") and m.group("num")) else "") + m.group("spec") if False else None
        # Попробуем простой перевод: если содержит 'f' или 'd' в конце — сохранить
        if s.endswith("f") or s.endswith("d"):
            return "%" + s
        # Иначе дефолт
        return "%s"

def find_py_files(path: str) -> List[str]:
    files = []
    if os.path.isfile(path) and path.endswith(".py"):
        files.append(path)
    elif os.path.isdir(path):
        for root, _, filenames in os.walk(path):
            for fn in filenames:
                if fn.endswith(".py"):
                    files.append(os.path.join(root, fn))
    return files

def process_file(path: str, inplace: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Возвращает (changed:bool, new_source_or_None)
    Если inplace=True и есть изменения — перезаписывает файл, делая резервную копию path + '.bak'.
    """
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    try:
        tree = ast.parse(src, filename=path)
    except Exception as e:
        print(f"[ERROR] parse {path}: {e}")
        return False, None

    transformer = FstringToLazyLoggingTransformer()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    new_src = astor.to_source(new_tree)

    if new_src != src:
        if inplace:
            bak = path + ".bak"
            with open(bak, "w", encoding="utf-8") as f:
                f.write(src)
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_src)
            print(f"[MODIFIED] {path}  (backup: {bak})")
        else:
            print(f"[WILL CHANGE] {path}")
        return True, new_src
    else:
        return False, None

def main():
    ap = argparse.ArgumentParser(description="Convert f-strings in logger calls to lazy logging format.")
    ap.add_argument("path", help="file or directory")
    ap.add_argument("--inplace", action="store_true", help="modify files in place (backup .bak)")
    args = ap.parse_args()

    files = find_py_files(args.path)
    if not files:
        print("No Python files found.")
        return

    changed = 0
    for f in files:
        ok, _ = process_file(f, inplace=args.inplace)
        if ok:
            changed += 1

    print(f"Done. Files changed: {changed}/{len(files)}")

if __name__ == "__main__":
    main()
