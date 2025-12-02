from TinyStatistician import TinyStatistician
import numpy as np
import math

EPS = 1e-9

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
    assert q == [2.75, 6.25], f"Even quartile failed: got {q}"
    print(f"✓ quartile({x2}) = {q}")
    
    # Odd length
    x3 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    q = TinyStatistician.quartile(x3)
    assert q == [3.0, 7.0], f"Odd quartile failed: got {q}"
    print(f"✓ quartile({x3}) = {q}")
    
    # Small array (4 elements)
    x4 = [1, 2, 3, 4]
    q = TinyStatistician.quartile(x4)
    assert q == [1.75, 3.25], f"4-element quartile failed: got {q}"
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
    
    # Single element (edge case)
    q = TinyStatistician.quartile([1])
    assert q == [1.0, 1.0], f"Single element failed: got {q}"
    print(f"✓ quartile([1]) = {q}")
    
    # Two elements (edge case)
    x7 = [1, 10]
    q = TinyStatistician.quartile(x7)
    assert q == [3.25, 7.75], f"2-element quartile failed: got {q}"
    print(f"✓ quartile({x7}) = {q}")
    
    # Negative numbers
    x8 = [-10, -5, 0, 5, 10, 15, 20]
    q = TinyStatistician.quartile(x8)
    assert q == [-2.5, 12.5], f"Negative quartile failed: got {q}"
    print(f"✓ quartile({x8}) = {q}")
    
    # Column vector
    x9 = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
    q = TinyStatistician.quartile(x9)
    assert q == [2.75, 6.25], f"Column vector quartile failed: got {q}"
    print(f"✓ quartile(column vector) = {q}")
    
    # Edge cases - should return None
    assert TinyStatistician.quartile([]) is None, "Empty quartile should return None"
    print("✓ quartile([]) = None")
    
    assert TinyStatistician.quartile(None) is None, "None quartile should return None"
    print("✓ quartile(None) = None")
    
    print()

def test_percentile():
    """Test percentile function."""
    print("=== TESTING PERCENTILE ===")
    
    # Basic test
    x1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = TinyStatistician.percentile(x1, 50)
    assert math.isclose(result, 5.5, rel_tol=EPS), f"50th percentile failed: got {result}"
    print(f"✓ percentile({x1}, 50) = {result}")
    
    # Edge percentiles
    result = TinyStatistician.percentile(x1, 0)
    assert math.isclose(result, 1.0, rel_tol=EPS), f"0th percentile failed: got {result}"
    print(f"✓ percentile({x1}, 0) = {result}")
    
    result = TinyStatistician.percentile(x1, 100)
    assert math.isclose(result, 10.0, rel_tol=EPS), f"100th percentile failed: got {result}"
    print(f"✓ percentile({x1}, 100) = {result}")
    
    # Quartiles via percentile
    x2 = [1, 42, 300, 10, 59]
    q1 = TinyStatistician.percentile(x2, 25)
    q3 = TinyStatistician.percentile(x2, 75)
    assert math.isclose(q1, 10.0, rel_tol=EPS) and math.isclose(q3, 59.0, rel_tol=EPS), f"Quartile via percentile failed: got Q1={q1}, Q3={q3}"
    print(f"✓ Q1={q1}, Q3={q3}")
    
    # Invalid percentile
    assert TinyStatistician.percentile(x1, -10) is None, "Negative percentile should fail"
    assert TinyStatistician.percentile(x1, 110) is None, "Percentile > 100 should fail"
    assert TinyStatistician.percentile(x1, "50") is None, "String percentile should fail"
    print("✓ Invalid percentile inputs handled")
    
    # Two elements (numpy accepts this)
    x3 = [1, 10]
    result = TinyStatistician.percentile(x3, 25)
    assert math.isclose(result, 3.25, rel_tol=EPS), f"2-element 25th percentile failed: got {result}"
    print(f"✓ percentile([1, 10], 25) = {result}")
    
    # Subject examples
    a = [1, 42, 300, 10, 59]
    result = TinyStatistician.percentile(a, 10)
    assert math.isclose(result, 4.6, rel_tol=EPS), f"10th percentile failed: got {result}"
    print(f"✓ percentile({a}, 10) = {result}")
    
    result = TinyStatistician.percentile(a, 15)
    assert math.isclose(result, 6.4, rel_tol=EPS), f"15th percentile failed: got {result}"
    print(f"✓ percentile({a}, 15) = {result}")
    
    result = TinyStatistician.percentile(a, 20)
    assert math.isclose(result, 8.2, rel_tol=EPS), f"20th percentile failed: got {result}"
    print(f"✓ percentile({a}, 20) = {result}")
    
    print()

def test_var():
    """Test variance function."""
    print("=== TESTING VARIANCE ===")
    
    # Subject example
    a = [1, 42, 300, 10, 59]
    result = TinyStatistician.var(a)
    assert math.isclose(result, 12279.439999999999, rel_tol=EPS), f"Subject variance failed: got {result}"
    print(f"✓ var({a}) = {result}")
    
    # Simple case
    x1 = [1, 2, 3, 4, 5]
    result = TinyStatistician.var(x1)
    # Population variance: mean=3, sum((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 5 = 10/5 = 2.0
    assert math.isclose(result, 2.0, rel_tol=EPS), f"Simple variance failed: got {result}"
    print(f"✓ var([1, 2, 3, 4, 5]) = {result}")
    
    # Two elements
    x2 = [1, 5]
    result = TinyStatistician.var(x2)
    # Population variance: mean=3, ((1-3)^2 + (5-3)^2) / 2 = 8/2 = 4.0
    assert math.isclose(result, 4.0, rel_tol=EPS), f"Two element variance failed: got {result}"
    print(f"✓ var([1, 5]) = {result}")
    
    # Single element (population variance should be 0)
    x3 = [42]
    result = TinyStatistician.var(x3)
    assert math.isclose(result, 0.0, rel_tol=EPS), f"Single element variance failed: got {result}"
    print(f"✓ var([42]) = {result}")
    
    # All same values
    x4 = [5, 5, 5, 5, 5]
    result = TinyStatistician.var(x4)
    assert math.isclose(result, 0.0, rel_tol=EPS), f"Zero variance failed: got {result}"
    print(f"✓ var([5, 5, 5, 5, 5]) = {result}")
    
    # Negative numbers
    x5 = [-5, -2, 0, 2, 5]
    result = TinyStatistician.var(x5)
    # mean=0, var = (25 + 4 + 0 + 4 + 25)/5 = 58/5 = 11.6
    assert math.isclose(result, 11.6, rel_tol=EPS), f"Negative variance failed: got {result}"
    print(f"✓ var([-5, -2, 0, 2, 5]) = {result}")
    
    # Column vector
    x6 = np.array([[1], [2], [3], [4], [5]])
    result = TinyStatistician.var(x6)
    assert math.isclose(result, 2.0, rel_tol=EPS), f"Column vector variance failed: got {result}"
    print(f"✓ var(column vector) = {result}")
    
    # Edge cases
    assert TinyStatistician.var([]) is None, "Empty variance should return None"
    print("✓ var([]) = None")
    
    assert TinyStatistician.var(None) is None, "None variance should return None"
    print("✓ var(None) = None")
    
    print()

def test_std():
    """Test standard deviation function."""
    print("=== TESTING STANDARD DEVIATION ===")
    
    # Subject example
    a = [1, 42, 300, 10, 59]
    result = TinyStatistician.std(a)
    assert math.isclose(result, 110.81263465868862, rel_tol=EPS), f"Subject std failed: got {result}"
    print(f"✓ std({a}) = {result}")
    
    # Simple case
    x1 = [1, 2, 3, 4, 5]
    result = TinyStatistician.std(x1)
    # std = sqrt(2.0) ≈ 1.4142135623730951
    assert math.isclose(result, math.sqrt(2.0), rel_tol=EPS), f"Simple std failed: got {result}"
    print(f"✓ std([1, 2, 3, 4, 5]) = {result}")
    
    # Two elements
    x2 = [1, 5]
    result = TinyStatistician.std(x2)
    # std = sqrt(4.0) = 2.0
    assert math.isclose(result, 2.0, rel_tol=EPS), f"Two element std failed: got {result}"
    print(f"✓ std([1, 5]) = {result}")
    
    # Single element (std should be 0)
    x3 = [42]
    result = TinyStatistician.std(x3)
    assert math.isclose(result, 0.0, rel_tol=EPS), f"Single element std failed: got {result}"
    print(f"✓ std([42]) = {result}")
    
    # All same values
    x4 = [5, 5, 5, 5, 5]
    result = TinyStatistician.std(x4)
    assert math.isclose(result, 0.0, rel_tol=EPS), f"Zero std failed: got {result}"
    print(f"✓ std([5, 5, 5, 5, 5]) = {result}")
    
    # Negative numbers
    x5 = [-5, -2, 0, 2, 5]
    result = TinyStatistician.std(x5)
    # std = sqrt(11.6) ≈ 3.4058772731852804
    assert math.isclose(result, math.sqrt(11.6), rel_tol=EPS), f"Negative std failed: got {result}"
    print(f"✓ std([-5, -2, 0, 2, 5]) = {result}")
    
    # Known std
    x6 = [2, 4, 4, 4, 5, 5, 7, 9]
    result = TinyStatistician.std(x6)
    # mean=5, var = (9+1+1+1+0+0+4+16)/8 = 32/8 = 4, std = 2
    assert math.isclose(result, 2.0, rel_tol=EPS), f"Known std failed: got {result}"
    print(f"✓ std([2, 4, 4, 4, 5, 5, 7, 9]) = {result}")
    
    # Column vector
    x7 = np.array([[1], [2], [3], [4], [5]])
    result = TinyStatistician.std(x7)
    assert math.isclose(result, math.sqrt(2.0), rel_tol=EPS), f"Column vector std failed: got {result}"
    print(f"✓ std(column vector) = {result}")
    
    # Edge cases
    assert TinyStatistician.std([]) is None, "Empty std should return None"
    print("✓ std([]) = None")
    
    assert TinyStatistician.std(None) is None, "None std should return None"
    print("✓ std(None) = None")
    
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
    print(f"percentile(a, 10) = {tstat.percentile(a, 10)} (expected: 4.6)")
    print(f"percentile(a, 15) = {tstat.percentile(a, 15)} (expected: 6.4)")
    print(f"percentile(a, 20) = {tstat.percentile(a, 20)} (expected: 8.2)")
    print(f"var(a) = {tstat.var(a)} (expected: 12279.439999999999)")
    print(f"std(a) = {tstat.std(a)} (expected: 110.81263465868862)")
    
    print()

if __name__ == "__main__":
    test_mean()
    test_median()
    test_quartile()
    test_percentile()
    test_var()
    test_std()
    test_subject_examples()
    print("✅ All tests passed!")
