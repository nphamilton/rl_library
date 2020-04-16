#!/bin/bash

cd cart_pole
python ars_cart_pole.py
python ddpg_cart_pole.py

cd ../inverted_pendulum
python ars_ip.py
python ddpg_ip.py

cd ..
python plot_results.py