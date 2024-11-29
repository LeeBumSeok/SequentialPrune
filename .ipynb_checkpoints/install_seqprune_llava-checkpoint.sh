conda create -n Seqprune_llava python=3.10 -y
conda activate Seqprune_llava
cd SequentialPrune/src
bash setup_llava.sh
cd ../eval/VLMEvalKit
pip install -e .