# python run_long_bench.py --model_name_or_path Llama-2-7b-hf_ratio-0.9_gs-1-fisher_uniform-whiten --lt_hadamard --lt_bits 4 --datasets repobench-p 
# python run_long_bench.py --model_name_or_path Llama-2-7b-hf_ratio-0.8_gs-1-fisher_uniform-whiten --lt_hadamard --lt_bits 4 --datasets repobench-p 
python run_long_bench.py --model_name_or_path ../Palu_custom_model/Llama-2-7b-hf_ratio-0.8_gs-1-fisher_uniform-whiten --lt_hadamard --lt_bits 3
# python run_long_bench.py --model_name_or_path ../Palu_custom_model/Llama-2-7b-hf_ratio-0.75_gs-1-fisher_uniform-whiten
