BMH running instruction:
compilation: nvcc -o bmh bmh.cu
usage: ./bmh target who pattern_mode pattern_path
target = path to the target text file
who = 0: sequential code on CPU, 1: Single-GPU execution, 2: Multi-GPU execution
pattern input mode = 0: read-in by command line, 1: read-in by file
pattern_path = (if choose file input) pattern file path

KMP running instruction:
compilation: nvcc -o kmp kmp.cu
usage: ./kmp shared_memory pinned_memory pattern_provided target_file pattern_file
shared_memory = 0: use global memory, 1: use shared memory 
pinned_memeory = 0: use pageable memory, 1: use pinned memory
pattern_provided = 0: from stdin, 1: from file
target_file: text file path
pattern_file: target file path (if 1 chosen for pattern_provided)
