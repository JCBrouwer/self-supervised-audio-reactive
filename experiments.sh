# python -m ssar.train --backbone gru --decoder fixed --hidden_size 3  # with fixed decoder, hidden size 3 translates to 17 intermediate envelopes
python -m ssar.train --backbone gru --decoder learned --hidden_size 16

python -m ssar.train --backbone gru --decoder fixed --hidden_size 3 --loss selfsupervised
python -m ssar.train --backbone gru --decoder learned --hidden_size 16 --loss selfsupervised

python -m ssar.train --backbone sashimi --decoder fixed --hidden_size 3 --n_examples 256_000
python -m ssar.train --backbone sashimi --decoder learned --hidden_size 16 --n_examples 256_000

python -m ssar.train --backbone sashimi --decoder fixed --hidden_size 3 --loss selfsupervised --n_examples 256_000
python -m ssar.train --backbone sashimi --decoder learned --hidden_size 16 --loss selfsupervised --n_examples 256_000
