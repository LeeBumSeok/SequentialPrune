conda create -n Seqprune_qwenvl2 python=3.10 -y
conda activate Seqprune_qwenvl2
cd SequentialPrune/src
bash setup_qwenvl2.sh
cd ../eval/VLMEvalKit
pip install -e .
cd ../../src
bash setup_qwenvl2.sh