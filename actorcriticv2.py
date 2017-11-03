import tensorflow as tf
import numpy as  np
from keras.models import Model
from keras.layers import Dense,Input,BatchNormalization,Concatenate
from keras.optimizers import Adam

class ActorNetwork:
	def __init__(self,state_dim,action_dim,lr,tau,gamma):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.lr =  lr
		self.tau = tau
		self.gamma =  gamma
		self.mainModel = self._build_model()
		self.targetModel = self._build_model()
'''
	def _build_model(self):
		model = Sequential()
		model.add(Dense(64,input_dim = self.state_dim, activation = 'relu'))
		model.add()
		model.add(Dense(64, activation = 'relu'))
		model.add(Dense(self.action_dim, activation = 'softmax'))
		model.compile(optimizer=Adam,loss='categorical_crossentropy',metrics=['accuracy'])
		return model
'''
	def _build_model(self):
		input_obs = Input(shape=(self.state_dim,))
		h = Dense(64)(input_obs)
		h = Activation('relu')(h)
		h = BatchNormalization()(h)
		h = Dense(64)(h)
		h = Activation('relu')(h)
		h = BatchNormalization()(h)
		h = Dense(10)(h)
		pred = Activation('softmax')(h)
		model = Model(inputs=input_obs,outputs=pred)
		model.compile(optimizer=Adam,loss='categorical_crossentropy')
		return model

	def act(self,state,noise):
		act = self.mainModel.predict(state) + noise
		return act

	def update_target(self):
		wMain =  np.asarray(self.mainModel.get_weights())
		wTarget = np.asarray(self.targetModel.get_weights())
		self.targetModel.set_weights(self.tau*wMain + (1.0-self.tau)*self.wTarget)


class CriticNetwork:
	def __init__(self,state_dim,actions_dim,lr,tau,gamma):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.lr =  lr
		self.tau = tau
		self.gamma =  gamma
		self.mainModel = self._build_model()
		self.targetModel = self._build_model()

	def _build_model(self):
		input_obs = Input(shape=(self.state_dim,))
		input_actions = Input(shape=(self.action_dim,))
		h = Dense(64)(input_obs)
		h = Activation('relu')(h)
		h = BatchNormalization()(h)
		h = Dense(64)(h)
		action_abs = Dense(64)(input_actions)
		h = keras.layers.concatenate([h,action_abs])
		h = Activation('relu')(h)
		h = BatchNormalization()(h)
		pred = Dense(1,kernel_initializer='random_uniform')(h)
		model = Model(inputs=[input_obs,input_actions],outputs=pred)
		model.compile(optimizer=Adam,loss='mean_squared_error')
		return model

	def update_target(self):
		wMain =  np.asarray(self.mainModel.get_weights())
		wTarget = np.asarray(self.targetModel.get_weights())
		self.targetModel.set_weights(self.tau*wMain + (1.0-self.tau)*self.wTarget)

	




