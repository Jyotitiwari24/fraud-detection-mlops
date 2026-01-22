import requests
import json
import pandas as pd

# API URL
API_URL = "http://127.0.0.1:8000"


def test_health():
    """Check if API is running"""
    print("="*60)
    print("TEST 1: Health Check")
    print("="*60)
    
    response = requests.get(f"{API_URL}/health")
    
    if response.status_code == 200:
        print("✓ API is running!")
        data = response.json()
        print(f"Model trained on: {data['model_info']['training_date']}")
        print(f"Model performance: {data['model_info']['metrics']}")
    else:
        print("✗ API is not responding")
    
    print()

def test_single_transaction():
    """Test with a single transaction"""
    print("="*60)
    print("TEST 2: Single Transaction Prediction")
    print("="*60)
    
    # Load real test data
    df = pd.read_csv('creditcard_data.csv')
    
    # Get a legitimate transaction
    legit_txn = df[df['Class'] == 0].iloc[0].to_dict()
    del legit_txn['Class']  # Remove the answer
    
    print("\nTesting LEGITIMATE transaction:")
    print(f"Amount: ${legit_txn['Amount']:.2f}")
    
    response = requests.post(f"{API_URL}/predict", json=legit_txn)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Result: {result['message']}")
        print(f"  Fraud probability: {result['fraud_probability']*100:.2f}%")
        print(f"  Risk level: {result['risk_level']}")
    else:
        print(f"✗ Error: {response.text}")
    
    # Get a fraudulent transaction
    fraud_txn = df[df['Class'] == 1].iloc[0].to_dict()
    del fraud_txn['Class']
    
    print("\n" + "-"*60)
    print("Testing FRAUDULENT transaction:")
    print(f"Amount: ${fraud_txn['Amount']:.2f}")
    
    response = requests.post(f"{API_URL}/predict", json=fraud_txn)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Result: {result['message']}")
        print(f"  Fraud probability: {result['fraud_probability']*100:.2f}%")
        print(f"  Risk level: {result['risk_level']}")
    else:
        print(f"✗ Error: {response.text}")
    
    print()


def test_batch_prediction():
    """Test batch prediction"""
    print("="*60)
    print("TEST 3: Batch Prediction")
    print("="*60)
    
    # Load test data
    df = pd.read_csv('creditcard_data.csv')
    
    # Get 10 random transactions
    sample = df.sample(n=10, random_state=42)
    transactions = []
    
    for _, row in sample.iterrows():
        txn = row.to_dict()
        del txn['Class']
        transactions.append(txn)
    
    print(f"Testing {len(transactions)} transactions...")
    
    response = requests.post(f"{API_URL}/predict/batch", json=transactions)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Batch prediction complete!")
        print(f"  Total transactions: {result['total_transactions']}")
        print(f"  Fraud detected: {result['fraud_detected']}")
        print(f"  Fraud rate: {result['fraud_rate']}%")
        
        # Show first few predictions
        print("\n  First 3 predictions:")
        for i, pred in enumerate(result['predictions'][:3], 1):
            status = "🚨 FRAUD" if pred['is_fraud'] else "✓ Legit"
            print(f"    {i}. {status} (confidence: {pred['fraud_probability']*100:.1f}%)")
    else:
        print(f"✗ Error: {response.text}")
    
    print()


def test_response_time():
    """Test API response time"""
    print("="*60)
    print("TEST 4: Response Time")
    print("="*60)
    
    import time
    
    # Load a transaction
    df = pd.read_csv('creditcard_data.csv')
    txn = df.iloc[0].to_dict()
    del txn['Class']
    
    # Test multiple times
    times = []
    num_tests = 100
    print(f"Making {num_tests} predictions to measure speed...")
    
    for i in range(num_tests):
        start = time.time()
        response = requests.post(f"{API_URL}/predict", json=txn)
        end = time.time()
        
        if response.status_code == 200:
            times.append((end - start) * 1000)  # Convert to milliseconds
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n✓ Speed test complete!")
        print(f"  Average response time: {avg_time:.2f}ms")
        print(f"  Fastest: {min_time:.2f}ms")
        print(f"  Slowest: {max_time:.2f}ms")
        
        if avg_time < 100:
            print("  👍 Excellent! < 100ms")
        elif avg_time < 500:
            print("  ✓ Good! < 500ms")
        else:
            print("  ⚠️ Slow. Consider optimization.")
    
    print()


# Run all tests
if   __name__ == "main":
    print("\n" + "="*60)
    print("FRAUD DETECTION API TESTING")
    print("="*60)
    print("\nMake sure the API is running first!")
    print("(In another terminal: uvicorn api:app --reload)")
    print()
    
    input("Press Enter to start tests...")
    
    try:
        test_health()
        test_single_transaction()
        test_batch_prediction()
        test_response_time()
        
        print("="*60)
        print("✓ ALL TESTS COMPLETE!")
        print("="*60)
        print("\nYour fraud detection API is working!")
        print("Next: Add monitoring and deploy to cloud")
        
    except requests.exceptions.ConnectionError:
        print("\n✗ ERROR: Cannot connect to API")
        print("Make sure the API is running:")
        print("  uvicorn api:app --reload")
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")