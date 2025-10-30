# Frontend Tests

This directory contains unit tests for the React frontend components.

## Setup

1. Install dependencies:
```bash
cd src/web_ui/frontend
npm install
```

2. Run tests:
```bash
npm test
```

## Test Structure

- `components/` - Component tests
- `stores/` - State management tests  
- `utils/` - Test utilities and helpers
- `setup.ts` - Global test setup

## Running Tests

### All tests:
```bash
npm test
```

### Watch mode:
```bash
npm run test:watch
```

### Coverage:
```bash
npm run test:coverage
```

### Specific test:
```bash
npm test -- Login.test.tsx
```

## Current Status

The test files are created but may have TypeScript compilation issues due to:
1. Missing type definitions for vitest globals
2. Import path resolution issues
3. Mock setup complexity

To run tests successfully, you may need to:
1. Install additional type definitions
2. Configure TypeScript paths properly
3. Simplify mock implementations

## Simple Test

A basic test (`simple.test.ts`) is included to verify the test runner works.