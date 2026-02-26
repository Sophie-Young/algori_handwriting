#SGD 随机梯度下降
#实现简单 但对所有参数使用相同的学习率 对于稀疏特征不友好 
class SGD:
    def __init__(self,lr):#lr一般取0.01
        self.lr=lr
    
    def update(self,params,grads):
        for key in params.keys():
            params[key]-=self.lr*grads[key]

#SGD 容易陷入局部最优 也就是在局部最低点的两边悬崖上来回跳转 但是出不去
#用小球模拟就是当他从山上滚下来的时候没有阻力的情况下 他会越滚越快，但是遇到了阻力就要慢慢走
#也就是说 加入了动量这一项以后 如果梯度方向不改变的话 参数更新速度很快 如果梯度方向改变的话 参数更新的速度就要减缓
class Momentum:
    def __init__(self,lr,momentum): #lr一般取0.01 momentum一般取0.9
        self.lr=lr
        self.momentum=momentum
        self.cache=None
    
    def update(self,params,grads):
        if self.cache is None:
            self.cache={}
            for key,val in params.items():
                self.cache[key]=np.zero_like(val)

        for key in params.keys():
            self.cache[key]*=momentum
            self.cache[key]-=self.lr*grads[key]
            params[key]+=self.cache[key]

#Adagrad 自适应学习率 对每个参数使用不同的学习率 Adaptive gradient algorithm
#对低频的参数使用较大的学习率 对高频的参数使用较小的学习率
class Adagrad:
    def __init__(self,lr,delta):
        self.lr=lr  #一般选取0.01
        self.cache=None
    
    def update(self,params,grads):
        if self.cache is None:
            self.cache={}
            for key in params.keys():
                self.cache[key]=np.zero_like(params[key])
        for key in params.keys():
            self.cache[key]+=grads[key]**2
            params[key]-=self.lr*grads[key]/(self.cache[key]+1e-7)**0.5

#Adagrad有一个问题 就是学习率会不断衰退 在很多参数取值还未达最优解时 学习率已经过小 从而导致无法收敛
#RMSProp使用指数衰减平均来慢慢迭起先前的梯度历史 防止学习率过早地过小
#RMSProp root mean square prop 梯度的均方根
class RMSProp:
    def __init__(self,lr,decay_rate):  # lr=0.01, decay_rate=0.99
        self.lr=lr
        self.decay_rate=decay_rate
        self.cache=None
    
    def update(self,params,grads):
        if self.cache is None:
            self.cache={}
            for key,val in params.items():
                self.cache[key]=np.zero_like(val)
        
        for key in params.keys():
            self.cache[key]*=self.decay_rate
            self.cache[key]+=(1-self.decay_rate)*grads[key]*grads[key]
            params[key]-=self.lr*grads[key]/(np.sqrt(self.cache[key])+1e-7)

#Adam Adaptive moment estimation
#对每个参数使用不同的学习率 融合了momentum和adaptive的优点 同时对梯度和学习率进行调整
#动量相当于增加参数更新的惯性 自适应过程相当于增加参数更新的阻力
class Adam:
    def __init__(self,lr=0.001,beta1=0.9,beta2=0.999):
        self.lr=lr
        self.beta1=beta1
        self.beta2=beta2
        self.m=0
        self.v=0
        self.iter=0

    def update(self,params,grads):
        if self.n is None:
            self.m , self.v={},{}
            for key , val in params.items():
                self.m[key]=np.zero_like(val)
                self.v[key]=np.zero_like(val)
        self.iter+=1

        lr_t=self.lr*np.sqrt(1.0-self.beta2**self.iter)/(1.0-self.beta1**self.iter)

        for key in params.keys():
            self.m[key]=self.m[key]*beta1+(1-beta1)*grads[key]
            self.v[key]=self.v[key]*beta2+(1-beta2)*grads[key]**2

            params[key]-=lr_t*self.m[key]/(np.sqrt(self.v[key])+1e-7)


                
                
