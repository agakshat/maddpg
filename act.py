from keras.models import load_model
import numpy as np
import make_env
import gym
from actorcriticv2 import ActorNetwork,CriticNetwork

actors = []
actors.append(load_model('actor0main0.h5'))
actors.append(load_model('actor1main0.h5'))
actors.append(load_model('actor2main0.h5'))

env = make_env.make_env('simple_spread')
while(1):
	for i in range(env.n):
		actor = actors[i]
		a.append(actor.act(np.reshape(s[i],(-1,actor.state_dim)),noise[i]()).reshape(actor.action_dim,))

	s2,r,done,_ = env.step(a) # a is a list with each element being an array
	env.render()