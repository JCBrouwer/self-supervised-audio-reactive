# common_args="--backbone gru --n_latent_split 3 --num_layers 4 --dropout 0.0 --duration 8 --fps 24 --n_examples 128_000 --lr 1e-4 --batch_size 32 --eval_every 10_240 --ckpt_every 10_240"
# python -m ssar.train $common_args --decoder learned --hidden_size 16 --loss supervised
# python -m ssar.train $common_args --decoder learned --hidden_size 16 --loss supervised --residual
# # with fixed decoder, hidden size 3 translates to 17 intermediate envelopes
# python -m ssar.train $common_args --decoder fixed --hidden_size 3 --loss supervised --residual
# python -m ssar.train $common_args --decoder fixed --hidden_size 3 --loss supervised
# python -m ssar.train $common_args --decoder learned --hidden_size 16 --loss selfsupervised
# python -m ssar.train $common_args --decoder learned --hidden_size 16 --loss selfsupervised --residual
# python -m ssar.train $common_args --decoder fixed --hidden_size 3 --loss selfsupervised --residual
# python -m ssar.train $common_args --decoder fixed --hidden_size 3 --loss selfsupervised

# common_args="--backbone gru --n_latent_split 3 --num_layers 6 --dropout 0.0 --duration 8 --fps 24 --n_examples 1_024_000 --lr 1e-4 --batch_size 32 --eval_every 102_400 --ckpt_every 102_400"
# python -m ssar.train $common_args --decoder learned --hidden_size 32 --loss supervised
# python -m ssar.train $common_args --decoder learned --hidden_size 32 --loss supervised --residual
# python -m ssar.train $common_args --decoder fixed --hidden_size 8 --loss supervised --residual
# python -m ssar.train $common_args --decoder fixed --hidden_size 8 --loss supervised
# python -m ssar.train $common_args --decoder learned --hidden_size 32 --loss selfsupervised
# python -m ssar.train $common_args --decoder learned --hidden_size 32 --loss selfsupervised --residual
# python -m ssar.train $common_args --decoder fixed --hidden_size 8 --loss selfsupervised --residual
# python -m ssar.train $common_args --decoder fixed --hidden_size 8 --loss selfsupervised

# common_args="--hidden_size 3 --loss selfsupervised --residual --n_latent_split 3 --dropout 0.0 --duration 8 --fps 24 --n_examples 256_000 --lr 1e-4 --batch_size 32 --eval_every 64_000 --ckpt_every 64_000"
# python -m ssar.train $common_args --decoder fixed --num_layers 2 --backbone gru
# python -m ssar.train $common_args --decoder fixed --num_layers 2 --backbone lstm
# python -m ssar.train $common_args --decoder fixed --num_layers 2 --backbone conv
# python -m ssar.train $common_args --decoder fixed --num_layers 2 --backbone mlp
# python -m ssar.train $common_args --decoder fixed --num_layers 2 --backbone transformer
# python -m ssar.train $common_args --decoder fixed --num_layers 2 --backbone sashimi
# python -m ssar.train $common_args --decoder fixed --num_layers 4 --backbone gru
# python -m ssar.train $common_args --decoder fixed --num_layers 4 --backbone lstm
# python -m ssar.train $common_args --decoder fixed --num_layers 4 --backbone conv
# python -m ssar.train $common_args --decoder fixed --num_layers 4 --backbone mlp
# python -m ssar.train $common_args --decoder fixed --num_layers 4 --backbone transformer
# python -m ssar.train $common_args --decoder fixed --num_layers 4 --backbone sashimi

# common_args="--hidden_size 3 --loss selfsupervised --residual --n_latent_split 3 --dropout 0.0 --duration 8 --fps 24 --n_examples 256_000 --lr 1e-4 --batch_size 32 --eval_every 64_000 --ckpt_every 64_000"
# python -m ssar.train $common_args --decoder fixed --num_layers 1 --backbone gru
# python -m ssar.train $common_args --decoder fixed --num_layers 8 --backbone gru
# python -m ssar.train $common_args --decoder fixed --num_layers 10 --backbone gru

# pip install git+https://github.com/NotNANtoN/lucid-sonic-dreams
for f in /home/hans/datasets/audiovisual/test_audio/*; do
    fn=${f##*/}
    echo python -c "from lucidsonicdreams import LucidSonicDream; LucidSonicDream(song=\"${f}\", style=\"modern art\").hallucinate(file_name=\"/home/hans/datasets/audiovisual/lucid/${fn%.*}.mp4\", fps=24, resolution=1024, batch_size=16)"
done
