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

common_args="--backbone gru --n_latent_split 3 --num_layers 6 --dropout 0.0 --duration 8 --fps 24 --n_examples 1_024_000 --lr 1e-4 --batch_size 32 --eval_every 102_400 --ckpt_every 102_400"

python -m ssar.train $common_args --decoder learned --hidden_size 32 --loss supervised
python -m ssar.train $common_args --decoder learned --hidden_size 32 --loss supervised --residual
python -m ssar.train $common_args --decoder fixed --hidden_size 8 --loss supervised --residual
python -m ssar.train $common_args --decoder fixed --hidden_size 8 --loss supervised

python -m ssar.train $common_args --decoder learned --hidden_size 32 --loss selfsupervised
python -m ssar.train $common_args --decoder learned --hidden_size 32 --loss selfsupervised --residual
python -m ssar.train $common_args --decoder fixed --hidden_size 8 --loss selfsupervised --residual
python -m ssar.train $common_args --decoder fixed --hidden_size 8 --loss selfsupervised
