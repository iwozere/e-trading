/**
 * Basic Frontend Tests
 * ------------------
 * 
 * Simple tests to verify the test setup works correctly.
 */

// Using vitest globals - no need to import

describe('Basic Frontend Tests', () => {
  it('should run basic assertions', () => {
    expect(1 + 1).toBe(2);
    expect('hello').toBe('hello');
    expect(true).toBeTruthy();
  });

  it('should work with mocks', () => {
    const mockFn = vi.fn();
    mockFn('test');
    
    expect(mockFn).toHaveBeenCalledWith('test');
    expect(mockFn).toHaveBeenCalledTimes(1);
  });

  it('should handle async operations', async () => {
    const asyncFn = vi.fn().mockResolvedValue('success');
    
    const result = await asyncFn();
    
    expect(result).toBe('success');
    expect(asyncFn).toHaveBeenCalled();
  });

  it('should mock fetch API', () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ data: 'test' }),
    });
    
    (globalThis as any).fetch = mockFetch;
    
    expect(fetch).toBeDefined();
    expect(typeof fetch).toBe('function');
  });

  it('should mock localStorage', () => {
    const mockStorage = {
      getItem: vi.fn(),
      setItem: vi.fn(),
      removeItem: vi.fn(),
      clear: vi.fn(),
    };
    
    Object.defineProperty(window, 'localStorage', {
      value: mockStorage,
    });
    
    window.localStorage.setItem('test', 'value');
    
    expect(mockStorage.setItem).toHaveBeenCalledWith('test', 'value');
  });
});