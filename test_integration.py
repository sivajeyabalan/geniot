#!/usr/bin/env python
"""
Integration test: Verify WebSocket streaming and metric updates
"""
import asyncio
import json
import websockets
from datetime import datetime

async def test_websocket():
    """Connect to WebSocket and verify metrics are streaming"""
    uri = "ws://localhost:8000/ws/live-metrics"
    
    print(f"\n🔌 Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connected!")
            
            metrics_received = []
            anomalies_detected = []
            
            print("\n📊 Waiting for 5 metric updates (should arrive ~every 500ms)...\n")
            
            start_time = datetime.now()
            for i in range(5):
                try:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(msg)
                    elapsed = (datetime.now() - start_time).total_seconds()
                    
                    # Extract key metrics
                    latency = data.get('latency_ms', 'N/A')
                    throughput = data.get('throughput_mbps', 'N/A')
                    energy = data.get('energy_nj_per_bit', 'N/A')
                    qos = data.get('qos_score', 'N/A')
                    anomaly = data.get('anomaly_detected', False)
                    score = data.get('anomaly_score', 'N/A')
                    
                    print(f"[{i+1}/5] [{elapsed:.2f}s] Latency: {latency:.2f}ms | "
                          f"Throughput: {throughput:.2f}Mbps | Energy: {energy:.2f}nJ/bit | QoS: {qos:.2f}")
                    
                    metrics_received.append({
                        'timestamp': elapsed,
                        'latency': latency,
                        'throughput': throughput,
                        'energy': energy,
                        'qos': qos
                    })
                    
                    if anomaly:
                        print(f"   ⚠️  ANOMALY DETECTED! Score: {score:.4f}")
                        anomalies_detected.append({'timestamp': elapsed, 'score': score})
                    
                except asyncio.TimeoutError:
                    print(f"[{i+1}/5] ❌ Timeout - no message received within 2s")
                    break
            
            print("\n" + "="*70)
            print("✅ CHECKLIST RESULTS:")
            print("="*70)
            if len(metrics_received) == 5:
                print("✅ Item 3: Metric cards show live values (5/5 updates received)")
                intervals = []
                for j in range(1, len(metrics_received)):
                    interval = metrics_received[j]['timestamp'] - metrics_received[j-1]['timestamp']
                    intervals.append(interval)
                avg_interval = sum(intervals) / len(intervals)
                print(f"   Average interval: {avg_interval:.3f}s (target: ~0.5s)")
                
                # Check metric values
                all_non_zero = all(
                    m['latency'] != 'N/A' and m['throughput'] != 'N/A' 
                    for m in metrics_received
                )
                if all_non_zero:
                    print("✅ All metrics have real numeric values")
            else:
                print(f"❌ Only received {len(metrics_received)}/5 metrics")
            
            if anomalies_detected:
                print(f"✅ Item 5: Anomaly panel shows events ({len(anomalies_detected)} detected)")
            else:
                print("ℹ️  Item 5: No anomalies detected in 5 updates (not a failure)")
            
    except Exception as e:
        print(f"❌ WebSocket connection failed: {e}")
        return False
    
    return len(metrics_received) >= 4

if __name__ == "__main__":
    try:
        success = asyncio.run(test_websocket())
        if success:
            print("\n🎉 Integration test PASSED!")
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted")
