/// <reference types="vitest" />

declare global {
  // Vitest globals
  const describe: typeof import('vitest').describe;
  const it: typeof import('vitest').it;
  const expect: typeof import('vitest').expect;
  const beforeEach: typeof import('vitest').beforeEach;
  const beforeAll: typeof import('vitest').beforeAll;
  const afterEach: typeof import('vitest').afterEach;
  const afterAll: typeof import('vitest').afterAll;
  const vi: typeof import('vitest').vi;
}