import numpy as np
import gym
import tensorflow as tf
import random
from ReplayMemory import ReplayMemory
#from actorcritic import ActorNetwork,CriticNetwork

def build_summaries():
	episode_reward = tf.Variable(0.)
	tf.summary.scalar("Reward",episode_reward)
	episode_ave_max_q = tf.Variable(0.)
	tf.summary.scalar("QMaxValue",episode_ave_max_q)
	summary_vars = [episode_reward,episode_ave_max_q]
	summary_ops = tf.summary.merge_all()
	return summary_ops, summary_vars

def train(self,sess,env,args,actors,critics,noise):

	summary_ops,summary_vars = build_summaries()
	init = tf.global_variables_initializer()
	sess.run(init)
	writer = tf.summary.FileWriter(args['summary_dir'],sess.graph)

	for actor in actors:
		actor.update_target()
	for critic in critics:
		critic.update_target()
	
	replayMemory = ReplayMemory(int(args['buffer_size']),int(args['random_seed']))

	for ep in range(int(args['max_episodes'])):

		s = env.reset()
		episode_reward = 0
		episode_av_max_q = 0

		for stp in range(int(args['max_episode_len'])):
			if args['render_env']:
				env.render()

			a = []
			for i in range(n):
				actor = actors[i]
				a.append(actor.act(np.reshape(s[i],(-1,actor.state_dim)),noise()).reshape(actor.output_dim,))
						
			s2,r,done,_ = env.step(a) # a is a nparray
			#replayMemory.add(np.reshape(s,(actor.input_dim,)),np.reshape(a,(actor.output_dim,)),r,done,np.reshape(s2,(actor.input_dim,)))
			replayMemory.add(s,a,r,done,s2)
			s = s2

			for i in range(n):
				actor = actors[i]
				critic = critics[i]
				if replayMemory.size()>int(args['minibatch_size']):

					s_batch,a_batch,r_batch,d_batch,s2_batch = replayMemory.miniBatch(int(args['minibatch_size']))
					a = []
					for j in range(n):
						s_batch_j = s_batch.transpose()[j]
						state_batch_j = np.asarray([x for x in s_batch_j])
						a.append(actors[j].predict_target(state_batch_j))

					s2_batch_i = np.asarray([x for x in s2_batch.transpose()[i]]) # Checked till this point, should be fine.
					targetQ = critic.predict_target(s2_batch_i,a) # Should  work, probbaly
					yi = []
					for k in range(int(args['minibatch_size'])):
						if d_batch.transpose()[i][k]:
							yi.append(r_batch.transpose()[i][k])
						else:
							yi.append(r_batch.transpose()[i][k] + critic.gamma*targetQ[k])
					
					predictedQValue = critic.train(s_batch,a_batch,yi)
					episode_av_max_q += np.amax(predictedQValue)
					
					actions_pred = []
					for j in range(n):
						s_batch_j = s2_batch.transpose()[j]
						state_batch_j = np.asarray([x for x in  s_batch_j])
						actions_pred.append(actors[j].predict(s_batch_j))

					#actions_pred = actor.predict(s_batch)
					grads = critic.actionGradients(s_batch,actions_pred)
					actor.train(s_batch,grads[0])
					
					actor.updateTargetNetwork()
					critic.updateTargetNetwork()

			episode_reward += r
			if done:
				summary_str = sess.run(summary_ops, feed_dict = {summary_vars[0]: episode_reward, summary_vars[1]: episode_av_max_q/float(stp)})
				writer.add_summary(summary_str,ep)
				writer.flush()
				print ('|Reward: {:d}| Episode: {:d}| Qmax: {:.4f}'.format(int(episode_reward),ep,(episode_av_max_q/float(stp))))
				break


