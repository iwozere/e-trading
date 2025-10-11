/**
 * Working Test Example
 * ------------------
 * 
 * Simple JavaScript test to verify the test setup works.
 * This avoids TypeScript configuration issues.
 */

describe('Working Test Suite', () => {
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

  it('should mock browser APIs', () => {
    // Test that our setup mocks are working
    expect(window.localStorage).toBeDefined();
    expect(window.matchMedia).toBeDefined();
    expect(globalThis.fetch).toBeDefined();
  });

  it('should handle DOM operations', () => {
    // Basic DOM test
    const div = document.createElement('div');
    div.textContent = 'Hello World';
    
    expect(div.textContent).toBe('Hello World');
    expect(div.tagName).toBe('DIV');
  });
});