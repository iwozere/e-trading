CBOE_URLS = {
    # Текущие файлы (обновляются ежедневно)
    "total_pc":  "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpc.csv",
    "equity_pc": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypc.csv",
    "index_pc":  "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/indexpc.csv",
    # Архивы
    "total_archive":  "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpcarchive.csv",
    "equity_archive": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypcarchive.csv",
    "index_archive":  "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/indexpcarchive.csv",
}

def download_cboe_pc(cache_dir: str = "data/sentiment") -> pd.DataFrame:
    frames = []
    for name, url in CBOE_URLS.items():
        r = requests.get(url, timeout=30)
        df = pd.read_csv(
            io.StringIO(r.text),
            skiprows=2,    # первые 2 строки — disclaimer
            parse_dates=["Trade Date"],
            na_values=["."]
        )
        df = df.rename(columns={"Trade Date": "date", "P/C Ratio": f"pc_{name}"})
        df = df[["date", f"pc_{name}"]].dropna()
        frames.append(df.set_index("date"))

    combined = pd.concat(frames, axis=1).sort_index()
    # Объединить архив + текущий в единый ряд
    combined["pc_equity"] = combined["pc_equity"].combine_first(combined["pc_equity_archive"])
    combined["pc_total"]  = combined["pc_total"].combine_first(combined["pc_total_archive"])

    combined.to_parquet(f"{cache_dir}/cboe_putcall.parquet")
    return combined