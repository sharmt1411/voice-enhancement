import os
import sys

# Ensure src is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from stream_larynx import StreamingArtificialLarynx

def main():
    print("=" * 60)
    print(" Software Artificial Larynx (软件版人工喉) Stream Benchmark")
    print("=" * 60)
    
    input_wav = os.path.join(os.path.dirname(__file__), "dataset", "audio_breath_9.wav")
    output_wav = os.path.join(os.path.dirname(__file__), "larynx_stream_output9.wav")
    
    print(f"Input audio: {input_wav}")
    print(f"Output audio: {output_wav}")
    
    larynx = StreamingArtificialLarynx()
    out_path, rtf, avg_latency = larynx.process_file_streaming(input_wav, output_wav, chunk_len_ms=500)
    
    print("\n--- Benchmark Results ---")
    print(f"Status: SUCCESS")
    print(f"End-to-End RTF: {rtf:.3f} (Requirement: RTF < 1.0)")
    print(f"Average Chunk Latency: {avg_latency:.1f}ms (Chunk Size: 500ms)")
    print(f"Saved to: {out_path}")
    assert rtf < 1.0, f"RTF {rtf:.3f} exceeds real-time threshold 1.0!"
    print("All real-time acoustic constraints PASSED!")

if __name__ == "__main__":
    main()
