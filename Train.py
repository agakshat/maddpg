import numpy as np
import gym
import tensorflow as tf
import random
from ReplayMemory import ReplayMemory

def build_summaries():
	episode_reward1 = tf.Variable(0.)
	tf.summary.scalar("Reward1",episode_reward1)
	episode_reward2 = tf.Variable(0.)
	tf.summary.scalar("Reward2",episode_reward2)
	summary_vars = [episode_reward1,episode_reward2]
	summary_ops = tf.summary.merge_all()

	return summary_ops, summary_vars

def train(sess,env,args,actors,critics,noise):

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
		episode_reward = np.zeros((env.n,))

		if ep%100==0:
			for k in range(env.n):
				file1 = 'results/actor'+str(k)+'/main'+str(ep)+'.h5'
				#file2 = 'results/actor'+str(k)+'/target'+str(ep)+'.h5'
				file3 = 'results/critic'+str(k)+'/main'+str(ep)+'.h5'
				#file4 = 'results/critic'+str(k)+'/target'+str(ep)+'.h5'
				actor = actors[k]
				critic = critics[k]
				actor.mainModel.save(file1)
				#actor.targetModel.save(file2)
				critic.mainModel.save(file3)
				#critic.targetModel.save(file4)

		for stp in range(int(args['max_episode_len'])):
			if args['render_env']:
				env.render()

			a = []
			for i in range(env.n):
				actor = actors[i]
				a.append(actor.act(np.reshape(s[i],(-1,actor.state_dim)),noise[i]()).reshape(actor.action_dim,))
			print("Action: {}".format(a))
			s2,r,done,_ = env.step(a) # a is a list with each element being an array
			replayMemory.add(s,a,r,done,s2)
			s = s2
			action_dims_done = 0
			for i in range(env.n):
				actor = actors[i]
				critic = critics[i]
				if replayMemory.size()>int(args['minibatch_size']):

					s_batch,a_batch,r_batch,d_batch,s2_batch = replayMemory.miniBatch(int(args['minibatch_size']))
					a = []
					for j in range(env.n):
						state_batch_j = np.asarray([x for x in s_batch[:,j]]) #batch processing will be much more efficient even though reshaping will have to be done
						a.append(actors[j].predict_target(state_batch_j))

					a_temp = np.transpose(np.asarray(a),(1,0,2))
					a_for_critic = np.asarray([x.flatten() for x in a_temp])
					s2_batch_i = np.asarray([x for x in s2_batch[:,i]]) # Checked till this point, should be fine.
					targetQ = critic.predict_target(s2_batch_i,a_for_critic) # Should  work, probably

					yi = []
					for k in range(int(args['minibatch_size'])):
						if d_batch[:,i][k]:
							yi.append(r_batch[:,i][k])
						else:
							yi.append(r_batch[:,i][k] + critic.gamma*targetQ[k])
					s_batch_i = np.asarray([x for x in s_batch[:,i]])
					critic.train(s_batch_i,np.asarray([x.flatten() for x in a_batch]),np.reshape(yi,(int(args['minibatch_size']),1)))
					
					actions_pred = []
					for j in range(env.n):
						state_batch_j = np.asarray([x for x in  s2_batch[:,j]])
						actions_pred.append(actors[j].predict(state_batch_j)) # Should work till here, roughly, probably

					a_temp = np.transpose(np.asarray(actions_pred),(1,0,2))
					a_for_critic_pred = np.asarray([x.flatten() for x in a_temp])
					s_batch_i = np.asarray([x for x in s_batch[:,i]])
					grads = critic.action_gradients(s_batch_i,a_for_critic_pred)[:,action_dims_done:action_dims_done + actor.action_dim]
					actor.train(s_batch_i,grads)
					actor.update_target()
					critic.update_target()
				
				action_dims_done = action_dims_done + actor.action_dim
			episode_reward += r
			#print(done)
			if np.all(done):
				#summary_str = sess.run(summary_ops, feed_dict = {summary_vars[0]: episode_reward, summary_vars[1]: episode_av_max_q/float(stp)})
				summary_str = sess.run(summary_ops, feed_dict = {summary_vars[0]: episode_reward[0], summary_vars[1]: episode_reward[3]})
				writer.add_summary(summary_str,ep)
				writer.flush()
				#print ('|Reward: {:d}| Episode: {:d}| Qmax: {:.4f}'.format(int(episode_reward),ep,(episode_av_max_q/float(stp))))
				print ('|Reward: {}	| Episode: {:d}'.format(episode_reward,ep))
				break

			if stp == int(args['max_episode_len'])-1:
				print ('|Reward: {}	| Episode: {:d}'.format(episode_reward,ep))


