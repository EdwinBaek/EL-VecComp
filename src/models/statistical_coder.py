import os
import csv
import heapq
import struct
import numpy as np
from collections import Counter

def quantize_vector(vector, compression_bits=8):
    min_val, max_val = np.min(vector), np.max(vector)
    step = (max_val - min_val) / (2 ** compression_bits - 1)
    quantized = np.round((vector - min_val) / step).astype(int)
    return quantized, min_val, max_val

def get_probabilities(data):
    counts = Counter(data)
    total = sum(counts.values())
    return {symbol: count / total for symbol, count in counts.items()}

# Arithmetic coding 계산을 위한 누적 분포 계산 함수
def get_cumulative_probs(probabilities):
    cumulative = {}
    total = 0
    for symbol, prob in sorted(probabilities.items()):
        cumulative[symbol] = (total, total + prob)
        total += prob
    return cumulative

def arithmetic_encode(data, probabilities):
    cumulative_probs = get_cumulative_probs(probabilities)
    low, high = 0.0, 1.0
    for symbol in data:
        range_width = high - low
        high = low + range_width * cumulative_probs[symbol][1]
        low = low + range_width * cumulative_probs[symbol][0]
    return (low + high) / 2

# Huffman Coding
class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(probabilities):
    heap = [Node(char, freq) for char, freq in probabilities.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        internal = Node(None, left.freq + right.freq)
        internal.left = left
        internal.right = right
        heapq.heappush(heap, internal)

    return heap[0]

def generate_huffman_codes(root):
    def traverse(node, code):
        if node.char is not None:
            return {node.char: code}
        left_codes = traverse(node.left, code + '0')
        right_codes = traverse(node.right, code + '1')
        return {**left_codes, **right_codes}

    return traverse(root, '')

def huffman_encode(data, codes):
    return ''.join(codes[symbol] for symbol in data)

def compress_vector(vector, coding_type, compression_bits=8):
    quantized, min_val, max_val = quantize_vector(vector, compression_bits)
    probabilities = get_probabilities(quantized)

    if coding_type == 'arithmetic':
        encoded = arithmetic_encode(quantized, probabilities)
        encoded_bytes = struct.pack('!d', encoded)
    elif coding_type == 'huffman':
        huffman_tree = build_huffman_tree(probabilities)
        huffman_codes = generate_huffman_codes(huffman_tree)
        encoded = huffman_encode(quantized, huffman_codes)
        encoded_bytes = int(encoded, 2).to_bytes((len(encoded) + 7) // 8, byteorder='big')
    else:
        raise ValueError("Invalid coding type. Choose 'arithmetic' or 'huffman'.")

    return {
        'encoded_bytes': encoded_bytes,
        'min_val': min_val,
        'max_val': max_val,
        'probabilities': probabilities
    }

def process_csv_file(input_file, output_file, coding_type, compression_bits=8):
    compressed_vectors = {}
    with open(input_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            identifier = row['identifier']
            vector = np.array([float(row[f'dim_{i}']) for i in range(128)])  # Assuming 128 dimensions
            compressed_vectors[identifier] = compress_vector(vector, coding_type, compression_bits)

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['identifier', 'encoded_bytes', 'min_val', 'max_val', 'probabilities'])
        for identifier, compressed_data in compressed_vectors.items():
            writer.writerow([
                identifier,
                compressed_data['encoded_bytes'].hex(),
                compressed_data['min_val'],
                compressed_data['max_val'],
                repr(compressed_data['probabilities'])
            ])

def main(input_dir, output_dir, coding_type='arithmetic', compression_bits=8):
    print(f"Coding type: {coding_type}")
    print(f"Compression bits: {compression_bits}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            print(f"Starting embedding compression from {filename}")
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, f"compressed_{filename}")
            print(f"Processing file: {filename}")
            process_csv_file(input_file, output_file, coding_type, compression_bits)
            print(f"Compressed file saved as: {output_file}")

    print("Embedding compression completed.")