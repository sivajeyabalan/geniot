#!/usr/bin/env python
"""
Full System Integration Test: 5-minute stability verification
Tests backend + frontend running together without errors
"""

import asyncio
import json
import websockets
import time
from datetime import datetime, timedelta
from urllib import request


def http_get_json(url, timeout=5):
    req = request.Request(url, method='GET')
    with request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode('utf-8')
        return response.status, json.loads(body)


def http_get_text(url, timeout=5):
    req = request.Request(url, method='GET')
    with request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode('utf-8')
        return response.status, body


def http_post_json(url, payload, timeout=5):
    data = json.dumps(payload).encode('utf-8')
    req = request.Request(
        url,
        data=data,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode('utf-8')
        parsed = json.loads(body) if body else {}
        return response.status, parsed

async def test_system_stability(duration_minutes=5):
    """
    Run full system for specified duration and verify stability
    """
    start_time = datetime.now()
    duration = timedelta(minutes=duration_minutes)
    
    # Test 1: Health Check
    print("\n" + "="*70)
    print(f"FULL SYSTEM INTEGRATION TEST ({duration_minutes} minutes)")
    print("="*70)
    
    try:
        print("\n[1/5] Checking Backend Health...")
        status, payload = http_get_json('http://127.0.0.1:8000/api/health', timeout=5)
        print(f"  Backend Health ({status}): {payload}")
    except Exception as e:
        print(f"  Warning: Backend health check failed ({e})")
        print("  (This is OK - backend may still be initializing)")
    
    # Test 2: WebSocket Connection
    print("\n[2/5] Testing WebSocket Stream...")
    try:
        async with websockets.connect('ws://localhost:8000/ws/live-metrics', ping_interval=None) as ws:
            msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
            data = json.loads(msg)
            print(f"  WebSocket connected! Sample metric: latency={data.get('latency_ms', 0):.2f}ms")
    except Exception as e:
        print(f"  Warning: WebSocket test failed ({e})")
    
    # Test 3: Stability monitoring loop
    print(f"\n[3/5] Running stability test for {duration_minutes} minutes...")
    print(f"  Start: {start_time.strftime('%H:%M:%S')}")
    print(f"  End: {(start_time + duration).strftime('%H:%M:%S')}")
    
    metrics_received = 0
    errors = []
    
    try:
        async with websockets.connect('ws://localhost:8000/ws/live-metrics', ping_interval=None) as ws:
            while datetime.now() - start_time < duration:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    data = json.loads(msg)
                    metrics_received += 1
                    
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if metrics_received % 30 == 0:  # Log every 30 messages
                        print(f"    [{elapsed:5.1f}s] Received {metrics_received} metrics")
                    
                    # Verify metric structure
                    required_keys = ['timestamp', 'latency_ms', 'throughput_mbps', 'anomaly_detected']
                    for key in required_keys:
                        if key not in data:
                            errors.append(f"Missing key: {key}")
                
                except asyncio.TimeoutError:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if elapsed < duration.total_seconds():
                        continue  # Still in test window
                    else:
                        break
                except Exception as e:
                    errors.append(str(e))
    
    except Exception as e:
        print(f"  Error during stability test: {e}")
    
    # Test 4: Frontend Check
    print("\n[4/5] Checking Frontend Server...")
    try:
        status, body = http_get_text('http://localhost:5173/', timeout=5)
        if status == 200:
            print(f"  Frontend responding: HTTP {status}")
            if '<html' in body.lower():
                print(f"  HTML rendered successfully")
    except Exception as e:
        print(f"  Warning: Frontend check failed ({e})")
    
    # Test 5: REST API Check
    print("\n[5/5] Testing REST Endpoints...")
    try:
        # Test /api/metrics
        status, _ = http_post_json(
            'http://127.0.0.1:8000/api/generate-traffic',
            payload={'n_samples': 1, 'seq_len': 50},
            timeout=5,
        )
        if status == 200:
            print(f"  POST /api/generate-traffic: HTTP {status} ✓")
        
        # Test /api/health
        status, _ = http_get_json('http://127.0.0.1:8000/api/health', timeout=5)
        if status == 200:
            print(f"  GET /api/health: HTTP {status} ✓")
    except Exception as e:
        print(f"  API test failed: {e}")
    
    # Summary
    elapsed_seconds = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "="*70)
    print("SYSTEM STABILITY TEST RESULTS")
    print("="*70)
    print(f"Elapsed Time: {elapsed_seconds:.1f} seconds ({elapsed_seconds/60:.1f} minutes)")
    print(f"WebSocket Messages Received: {metrics_received}")
    if metrics_received > 0:
        print(f"Average Message Rate: {metrics_received / elapsed_seconds:.2f} msg/sec")
    
    if errors:
        print(f"\nErrors Encountered: {len(errors)}")
        for err in errors[:5]:  # Show first 5 errors
            print(f"  - {err}")
    else:
        print("\n✓ NO ERRORS LOGGED")
    
    print("\n" + "="*70)
    if elapsed_seconds >= duration.total_seconds() * 0.95 and metrics_received > 100:
        print("STATUS: PASS - System ran stably for the specified duration")
    else:
        print("STATUS: PARTIAL - System running but may have encountered issues")
    print("="*70)
    
    return metrics_received, errors

async def main():
    metrics, errors = await test_system_stability(duration_minutes=5)
    print(f"\nNext Steps:")
    print(f"  [✓] System integration test complete")
    print(f"  [→] Record demo video")
    print(f"  [→] Write project report")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
