# Perturbation Parameters

| Parameter | Value | Meaning |
| --- | --- | --- |
| padding_values | 0, 16, 64 bytes | Extra TCP payload bytes appended to client-to-server packets |
| delay_values_ms | 0, 50, 100 ms | Timing offsets applied after the first packet in each flow |
| min_baseline_detection | 0.5 | Warning threshold for original PCAP detection quality |
| min_peak_rate_ratio | 0.5 | Minimum SYN peak-rate retention for a valid variant |
| seed | 42 | Fixed seed for split and evaluation sampling |

## Invariant checks

- ip_proto
- tp_dst
- packet_count
- syn_count
- ack_count
- rst_count
- fin_count
- psh_count
- 5-tuple-like flow identity fields

## Derived values

- flow_duration = flow_duration_sec + flow_duration_nsec / 1e9
- byte_count = frame length for the packet payload summary
- packet_count_per_second = packet_count / flow_duration
- byte_count_per_second = byte_count / flow_duration
