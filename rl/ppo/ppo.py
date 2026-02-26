from network import FeedForwardNN
import torch
from torch.distributions import  MultivariateNormal
from torch.optim import Adam
import torch.nn as nn 
import numpy as np
import gym

class PPO:
    def __init__(self,env):
        #Initialize hyperparameters
        self._init_hyperparameters()

        #Extract environment information 
        self.env=env
        self.obs_dim=env.observation_space.shape[0]
        self.act_dim=env.action_space.shape[0]

        #Initialize actor and critic networks
        self.actor=FeedForwardNN(self.obs_dim,self.act_dim)
        self.critic=FeedForwardNN(self.obs_dim,1)

        #Initialize actor optimizer
        self.actor_optim=Adam(self.actor.parameters(),lr=self.lr)      

        #Initialize critic optimizer
        self.critic_optim=Adam(self.critic.parameters(),lr=self.lr)   
        
        # create the covariance matrix for get action 
        self.cov_var=torch.full(size=(self.act_dim,),fill_value=0.5)
        self.cov_mat=torch.diag(self.cov_var)

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode=1600
        self.gemma=0.95
        self.n_updates_per_itertion=5
        self.clip = 0.2 # As recommended by the paper
        self.lr = 0.005

    def get_action(self,obs):
        # query the actor network for a mean action
        #same thing as calling self.actor.forward(obs)

        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        mean=self.actor(obs)
        dist=MultivariateNormal(mean,self.cov_mat)
        action=dist.sample()
        log_prob=dist.log_prob(action)
        return action.detach().numpy(),log_prob.detach()


    def compute_rtgs(self,batch_rewards):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs=[]

        #iterate through each episode backwards to maintain same order

        for ep_rewards in reversed(batch_rewards):
            discounted_reward=0

            for rewards in reversed(ep_rewards):
                discounted_reward = discounted_reward * self.gemma + rewards
                batch_rtgs.insert(0,discounted_reward)
            
        batch_rtgs=torch.tensor(batch_rtgs,dtype=torch.float)

        return batch_rtgs
    
    def rollout(self):
        batch_obs=[]
        batch_acts=[]
        batch_log_probs=[]
        batch_rewards=[]
        batch_returns_to_gos=[]
        batch_len=[] #episodic lengths in batch 

        t=0 # timesteps simulated so far

        while t < self.timesteps_per_batch:

            #rewards this episode
            ep_rewards=[]

            obs,info=self.env.reset()
            done=False

            for ep_t in range(self.max_timesteps_per_episode):
                t+=1

                #collect observation
                batch_obs.append(obs)

                action,log_prob=self.get_action(obs)
                obs,reward,terminated,truncated,_ =self.env.step(action)
                done=terminated or truncated

                #collect reward action log prob
                ep_rewards.append(reward)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break
            
            # collect episodic length and rewards
            batch_len.append(ep_t+1)
            batch_rewards.append(ep_rewards)

        # reshape data as tensors in the shape specified before returning 
        batch_obs=torch.tensor(batch_obs,dtype=torch.float)
        batch_acts=torch.tensor(batch_acts,dtype=torch.float)
        batch_log_probs=torch.tensor(batch_log_probs,dtype=torch.float)

        batch_returns_to_gos=self.compute_rtgs(batch_rewards)

        #return the batch data
        return batch_obs,batch_acts,batch_log_probs,batch_returns_to_gos,batch_len
    

    
    def rollout4gae(self,t_so_far):
        batch_obs=[]
        batch_acts=[]
        batch_log_probs=[]
        batch_rewards=[]
        batch_values=[]
        batch_len=[] #episodic lengths in batch 
        batch_dones=[]

        ep_rewards=[]
        ep_done=[]
        ep_values=[]

        t=0 # timesteps simulated so far

        while t < self.timesteps_per_batch:


            ep_rewards=[]
            ep_dones=[]
            ep_values=[]

            obs,info=self.env.reset()
            done=False

            for ep_t in range(self.max_timesteps_per_episode):
                if(self.render):
                    self.env.render()
                ep_dones.append(done)

                t+=1

                #collect observation
                batch_obs.append(obs)

                action,log_prob=self.get_action(obs)
                val=self.critic(obs)

                obs,reward,terminated,truncated,_ =self.env.step(action)
                done=terminated or truncated

                #collect reward action log prob
                ep_rewards.append(reward)
                ep_values.append(val.flatten())
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break
            
            # collect episodic length and rewards
            batch_len.append(ep_t+1)
            batch_rewards.append(ep_rewards)
            batch_values.append(ep_values)
            batch_dones.append(ep_dones)

        # reshape data as tensors in the shape specified before returning 
        batch_obs=torch.tensor(batch_obs,dtype=torch.float)
        batch_acts=torch.tensor(batch_acts,dtype=torch.float)
        batch_log_probs=torch.tensor(batch_log_probs,dtype=torch.float)


        #return the batch data
        return batch_obs,batch_acts,batch_log_probs,batch_rewards,batch_len,batch_values,batch_dones
    


 
    def learn(self,total_timesteps):
        t_so_far=0 #timesteps simulated so far
        while t_so_far < total_timesteps:
            batch_obs,batch_acts,batch_log_probs,batch_rtgs,batch_lens=self.rollout()
            #batch_obs,batch_acts,batch_log_probs,batch_rewards,batch_len,batch_values,batch_dones=self.rollout(t_so_far)

            # Calculate how many timesteps we collected this batch 
            t_so_far += np.sum(batch_lens)

            #calculate V_{pai,k}
            V,_=self.evaluate(batch_obs,batch_acts)
            #V=self.crirtic(batch_obs).squeuze()


            #calculate advantages
            A_k=batch_rtgs-V.detach()
            #A_k=self.calculate_gae(batch_rewards,batch_values,batch_dones)
            #batch_rtgs=A_k+V.detach()

            #trick!! make training more stable
            A_k=(A_k-A_k.mean())/(A_k.std()+1e-10)

            for _ in range(self.n_updates_per_itertion):
                frac=(t_so_far-1.0)/total_timesteps
                new_lr=self.lr*(1.0-frac)
                new_lr=max(new_lr,1.0)
                self.actor_optim.param_groups[0]["lr"]=new_lr
                self.critic_optim.param_groups[0]["lr"]=new_lr

                #calculate pi_theta(a_t | s_t)
                V,curr_log_probs=self.evaluate(batch_obs,batch_acts)    

                #calculate ratios
                ratios=torch.exp(curr_log_probs-batch_log_probs)    

                #calculate surrogate losses
                surr1=ratios*A_k
                surr2=torch.clamp(ratios,1-self.clip,1+self.clip)*A_k

                actor_loss=(-torch.min(surr1,surr2)).mean()
                critic_loss=nn.MSELoss()(V,batch_rtgs)

                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

    def learn_minibatch(self,total_timesteps):
        t_so_far=0 #timesteps simulated so far
        while t_so_far < total_timesteps:
            batch_obs,batch_acts,batch_log_probs,batch_rtgs,batch_lens=self.rollout()

            # Calculate how many timesteps we collected this batch 
            t_so_far += np.sum(batch_lens)

            #calculate V_{pai,k}
            V,_=self.evaluate(batch_obs,batch_acts)

            #calculate advantages
            A_k=batch_rtgs-V.detach()

            #trick!! make training more stable
            A_k=(A_k-A_k.mean())/(A_k.std()+1e-10)

            step = batch_obs.size(0)
            index = np.arange(step)
            minibatch_size = step//self.num_minibatches

            
            for _ in range(self.n_updates_per_itertion):
                frac=(t_so_far-1.0)/total_timesteps
                new_lr=self.lr*(1.0-frac)
                new_lr=max(new_lr,1.0)
                self.actor_optim.param_groups[0]["lr"]=new_lr
                self.critic_optim.param_groups[0]["lr"]=new_lr

                np.random.shuffle(index)
                for start in range(0,step,minibatch_size):
                    end=start+minibatch_size
                    idx=index[start:end]
                    mini_obs=batch_obs[idx]
                    mini_acts=batch_acts[idx]
                    mini_log_probs=batch_log_probs[idx]
                    mini_advantage=A_k[idx]
                    mini_rtgs=batch_rtgs[idx]

                    #calculate pi_theta(a_t | s_t)
                    V,curr_log_probs=self.evaluate(mini_obs,mini_acts)    

                    #calculate ratios
                    ratios=torch.exp(curr_log_probs-mini_log_probs)    

                    #calculate surrogate losses
                    surr1=ratios*mini_advantage
                    surr2=torch.clamp(ratios,1-self.clip,1+self.clip)*mini_advantage

                    actor_loss=(-torch.min(surr1,surr2)).mean()
                    critic_loss=nn.MSELoss()(V,mini_rtgs)

                    self.actor_optim.zero_grad()
                    actor_loss.backward()
                    self.actor_optim.step()

                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    self.critic_optim.step()

    def calculate_gae(self,rewards,values,dones):
        batch_advantage=[]
        for ep_rewards,ep_values,ep_dones in zip(rewards,values,dones):
            advantages=[]
            last_advantage=0

            for t in reversed(range(len(ep_rewards))):
                if t+1 < len(ep_rewards):
                    delta=ep_rewards[t]+self.gemma*ep_values[t+1]*(1-ep_dones[t+1])-ep_values[t]
                else:
                    delta=ep_rewards[t]-ep_values[t]
                
                advantage=delta+self.gemma*self.lam*(1-ep_dones[t])*last_advantage
                last_advantage=advantage
                advantages.insert(0,advantage)

            batch_advantage.append(advantages)
        
        return torch.tensor(batch_advantage,dtype=torch.float)


    def evaluate(self,batch_obs,batch_acts):
        #query critic network for a value v for each obs in batch_obs.
        V = self.critic(batch_obs).squeeze()

        mean=self.actor(batch_obs)
        dist=MultivariateNormal(mean,self.cov_mat)
        log_probs=dist.log_prob(batch_acts)
        return V,log_probs
    
env = gym.make('Pendulum-v1')
model = PPO(env)
model.learn(10000)
