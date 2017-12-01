import tensorflow as tf
import numpy as np
import make_env
import gym
from keras.models import load_model
from ExplorationNoise import OrnsteinUhlenbeckActionNoise as OUNoise
import time

actors = []
actors.append(load_model('results/actor0/main16000.h5'))
actors.append(load_model('results/actor1/main16000.h5'))
actors.append(load_model('results/actor2/main16000.h5'))

env = make_env.make_env('simple_spread')
s = env.reset()
while(1):
	a = []
	for i in range(env.n):
		actor = actors[i]
		noise = OUNoise(mu = np.zeros(5))
		a.append((actor.predict(np.reshape(s[i],(-1,actor.input_shape[1])))+noise()).reshape(actor.output_shape[1],))

	s2,r,done,_ = env.step(a) # a is a list with each element being an array
	env.render()
	s = s2
	print("next episode")
	if np.all(done):
		s = env.reset()
	time.sleep(0.2)