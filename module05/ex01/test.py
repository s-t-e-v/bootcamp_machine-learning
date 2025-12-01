from TinyStatistician import TinyStatistician
import numpy as np

def test_mean():
    """Test mean function with various inputs."""
    print("=== TESTING MEAN ===")
    
    # Basic test
    x1 = [1, 42, 300, 10, 59]
    assert TinyStatistician.mean(x1) == 82.4, "Basic mean failed"
    print(f"✓ mean({x1}) = {TinyStatistician.mean(x1)}")
    
    # Column vector
    x2 = np.array([[1], [2], [3], [4], [5]])
    assert TinyStatistician.mean(x2) == 3.0, "Column vector mean failed"
    print(f"✓ mean(column vector) = {TinyStatistician.mean(x2)}")
    
    # Single element
    x3 = [42]
    assert TinyStatistician.mean(x3) == 42.0, "Single element failed"
    print(f"✓ mean([42]) = {TinyStatistician.mean(x3)}")
    
    # Negative numbers
    x4 = [-5, -2, 0, 2, 5]
    assert TinyStatistician.mean(x4) == 0.0, "Negative numbers failed"
    print(f"✓ mean({x4}) = {TinyStatistician.mean(x4)}")
    
    # Floats
    x5 = [1.5, 2.5, 3.5]
    assert abs(TinyStatistician.mean(x5) - 2.5) < 1e-10, "Float mean failed"
    print(f"✓ mean({x5}) = {TinyStatistician.mean(x5)}")
    
    # Edge cases - should return None
    assert TinyStatistician.mean([]) is None, "Empty list should return None"
    print("✓ mean([]) = None")
    
    assert TinyStatistician.mean(None) is None, "None input should return None"
    print("✓ mean(None) = None")
    
    assert TinyStatistician.mean("not a list") is None, "String should return None"
    print("✓ mean('not a list') = None")
    
    assert TinyStatistician.mean([[1, 2], [3, 4]]) is None, "2D array should return None"
    print("✓ mean(2D array) = None")
    
    # NaN/Inf rejection
    assert TinyStatistician.mean([1, 2, np.nan]) is None, "NaN should be rejected"
    print("✓ mean([1, 2, nan]) = None")
    
    assert TinyStatistician.mean([1, 2, np.inf]) is None, "Inf should be rejected"
    print("✓ mean([1, 2, inf]) = None")
    
    print()

def test_median():
    """Test median function with various inputs."""
    print("=== TESTING MEDIAN ===")
    
    # Odd length
    x1 = [1, 42, 300, 10, 59]
    assert TinyStatistician.median(x1) == 42.0, "Odd length median failed"
    print(f"✓ median({x1}) = {TinyStatistician.median(x1)}")
    
    # Even length
    x2 = [1, 2, 3, 4]
    assert TinyStatistician.median(x2) == 2.5, "Even length median failed"
    print(f"✓ median({x2}) = {TinyStatistician.median(x2)}")
    
    # Unsorted data (odd)
    x3 = [2, 3, 2, 5, 9, 1]
    assert TinyStatistician.median(x3) == 2.5, "Unsorted odd median failed"
    print(f"✓ median({x3}) = {TinyStatistician.median(x3)}")
    
    # Unsorted data (even)
    x4 = [4, 5, 1, 0, 10]
    assert TinyStatistician.median(x4) == 4.0, "Unsorted even median failed"
    print(f"✓ median({x4}) = {TinyStatistician.median(x4)}")
    
    # Single element
    x5 = [42]
    assert TinyStatistician.median(x5) == 42.0, "Single element median failed"
    print(f"✓ median([42]) = {TinyStatistician.median(x5)}")
    
    # Two elements
    x6 = [1, 5]
    assert TinyStatistician.median(x6) == 3.0, "Two element median failed"
    print(f"✓ median({x6}) = {TinyStatistician.median(x6)}")
    
    # Negative numbers
    x7 = [-10, -5, 0, 5, 10]
    assert TinyStatistician.median(x7) == 0.0, "Negative median failed"
    print(f"✓ median({x7}) = {TinyStatistician.median(x7)}")
    
    # Column vector
    x8 = np.array([[1], [2], [3], [4], [5]])
    assert TinyStatistician.median(x8) == 3.0, "Column vector median failed"
    print(f"✓ median(column vector) = {TinyStatistician.median(x8)}")
    
    # Edge cases
    assert TinyStatistician.median([]) is None, "Empty median should return None"
    print("✓ median([]) = None")
    
    assert TinyStatistician.median(None) is None, "None median should return None"
    print("✓ median(None) = None")
    
    print()

def test_quartile():
    """Test quartile function with various inputs."""
    print("=== TESTING QUARTILE ===")
    
    # Basic test
    x1 = [1, 42, 300, 10, 59]
    q = TinyStatistician.quartile(x1)
    assert q == [10.0, 59.0], f"Basic quartile failed: got {q}"
    print(f"✓ quartile({x1}) = {q}")
    
    # Even length
    x2 = [1, 2, 3, 4, 5, 6, 7, 8]
    q = TinyStatistician.quartile(x2)
    assert q == [2.5, 6.5], f"Even quartile failed: got {q}"
    print(f"✓ quartile({x2}) = {q}")
    
    # Odd length
    x3 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    q = TinyStatistician.quartile(x3)
    assert q == [2.5, 7.5], f"Odd quartile failed: got {q}"
    print(f"✓ quartile({x3}) = {q}")
    
    # Small array (4 elements)
    x4 = [1, 2, 3, 4]
    q = TinyStatistician.quartile(x4)
    assert q == [1.5, 3.5], f"4-element quartile failed: got {q}"
    print(f"✓ quartile({x4}) = {q}")
    
    # 5 elements
    x5 = [1, 2, 3, 4, 5]
    q = TinyStatistician.quartile(x5)
    assert q == [2.0, 4.0], f"5-element quartile failed: got {q}"
    print(f"✓ quartile({x5}) = {q}")
    
    # Unsorted
    x6 = [9, 1, 5, 3, 7]
    q = TinyStatistician.quartile(x6)
    assert q == [3.0, 7.0], f"Unsorted quartile failed: got {q}"
    print(f"✓ quartile({x6}) = {q}")
    
    # Two elements (edge case)
    x7 = [1, 10]
    q = TinyStatistician.quartile(x7)
    assert q == [1.0, 10.0], f"2-element quartile failed: got {q}"
    print(f"✓ quartile({x7}) = {q}")
    
    # Negative numbers
    x8 = [-10, -5, 0, 5, 10, 15, 20]
    q = TinyStatistician.quartile(x8)
    assert q == [-5.0, 10.0], f"Negative quartile failed: got {q}"
    print(f"✓ quartile({x8}) = {q}")
    
    # Column vector
    x9 = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
    q = TinyStatistician.quartile(x9)
    assert q == [2.5, 6.5], f"Column vector quartile failed: got {q}"
    print(f"✓ quartile(column vector) = {q}")
    
    # Edge cases - should return None
    assert TinyStatistician.quartile([]) is None, "Empty quartile should return None"
    print("✓ quartile([]) = None")
    
    assert TinyStatistician.quartile([1]) is None, "Single element should return None"
    print("✓ quartile([1]) = None")
    
    assert TinyStatistician.quartile(None) is None, "None quartile should return None"
    print("✓ quartile(None) = None")
    
    print()

def test_subject_examples():
    """Test examples from the subject."""
    print("=== TESTING SUBJECT EXAMPLES ===")
    
    tstat = TinyStatistician()
    a = [1, 42, 300, 10, 59]
    
    print(f"a = {a}")
    print(f"mean(a) = {tstat.mean(a)} (expected: 82.4)")
    print(f"median(a) = {tstat.median(a)} (expected: 42.0)")
    print(f"quartile(a) = {tstat.quartile(a)} (expected: [10.0, 59.0])")
    
    print()

if __name__ == "__main__":
    test_mean()
    test_median()
    test_quartile()
    test_subject_examples()
    print("✅ All tests passed!")
