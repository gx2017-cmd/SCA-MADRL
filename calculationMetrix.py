import seaborn as sns;
# sns.set()
sns.set(style="white", palette="muted", color_codes=True,font_scale=2.6)
sns.palplot(sns.color_palette('colorblind',12))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.rc('font',family='Times New Roman')
plt.rcParams.update({'font.size':24,'font.weight':'bold'})

#Smooth the reward curve when plotting.
def smooth(data, sm=1):
    smooth_data = []
    if sm > 1:
        for d in data:
            z = np.ones(len(d))
            y = np.ones(sm)*1.0
            d = np.convolve(y, d, "same")/np.convolve(y, z, "same")
            smooth_data.append(d)
    return smooth_data

#Calculates the evaluation metric value of the training set.
def get_train_metric(success_rate_2=400):
    result_data=[]
    for i in ['0','2','4']:
        j=i
        for model in ['dqn','ddqn','double_drqn','dueling_ddqn']:
            item=[]
            try:
                one_m=pd.read_csv('result/single/%s_%s.csv'%(model,j))
            except:
                one_m = pd.read_csv('result/single/%s_%s.csv' % (model, j),encoding='gb18030')
            one_1=one_m['rewardAC'].mean() #AR
            one_2 = np.mean(np.sqrt(one_m['盾首水平偏差']**2+one_m['盾首垂直偏差']**2))  # FAD
            one_3 = one_m['rewardAC'].max()  # Max
            one_4 = one_m['rewardAC'].min()  # Min
            one_5 = np.std(one_m['rewardAC'][-500:]/one_m['rewardAC'].max()) # Std
            one_6 = one_m[abs(one_m['rewardAC']) >= success_rate_2]['盾首水平偏差'].count()/10  # T1
            one_7 = one_m[abs(one_m['rewardAC']) >= success_rate_2]['盾首水平偏差'].count()/10 # T2
            one_8 = one_m[abs(one_m['rewardAC']) >= success_rate_2+200]['盾首水平偏差'].count()/10  # TS
            list=one_m['rewardAC'].to_list()
            average_list = []
            for index in range(len(list)):
                if index<900:
                    li = list[index:index+100]
                else:
                    li=list[index:]
                average_list.append(np.array(li).mean())
            one_9=0
            for  x in range(30,len(average_list)):
                if x<900:
                    if np.std(one_m['rewardAC'][x:x+100] / one_m['rewardAC'][x:].max())<2*one_5:
                        one_9=average_list[x]/x
                        break
                else:
                    if np.std(one_m['rewardAC'][x:] / one_m['rewardAC'][x:].max()) < 2*one_5:
                        one_9 = average_list[x] / x  #SI
                        break
            item.append('1_'+model)
            item.append(j)
            item.append(one_1)
            item.append(one_2)
            item.append(one_3)
            item.append(one_4)
            item.append(one_5)
            item.append(one_6)
            item.append(one_7)
            item.append(one_8)
            item.append(one_9)
            result_data.append(item)
            print('model:%s         ,seed:%s'%(model,j))
        for model in ['dqn','ddqn','double_drqn','dueling_ddqn']:
            j = i
            item=[]
            try:
                one_m = pd.read_csv('result/double/%s_%s.csv' % (model, j))
            except:
                one_m = pd.read_csv('result/double/%s_%s.csv' % (model, j), encoding='gb18030')
            one_1=np.mean(one_m['rewardAC']+one_m['rewardBD'])#AR
            one_2 = np.mean(np.sqrt(one_m['盾首水平偏差']**2+one_m['盾首垂直偏差']**2))  # FAD
            one_3 = np.max(one_m['rewardAC']+one_m['rewardBD'])  # Max
            one_4 = np.min(one_m['rewardAC'] + one_m['rewardBD'])  # Min
            one=one_m['rewardAC']+one_m['rewardBD']
            one_5 = np.std(one[-500:]/one.max())  # Std
            one_6 = one_m[abs(one_m['rewardAC']) >= success_rate_2]['盾首水平偏差'].count()/10  # T1
            one_7 = one_m[abs(one_m['rewardBD']) >= success_rate_2]['盾首水平偏差'].count()/10  # T2
            one_8 = one_m[(abs(one_m['rewardAC']) >= success_rate_2-100) & (abs(one_m['rewardBD']) >= success_rate_2-100)]['盾首水平偏差'].count()/10  # TS
            list = one.to_list()
            average_list = []
            for index in range(len(list)):
                if index < 900:
                    li = list[index:index + 100]
                else:
                    li = list[index:]
                average_list.append(np.array(li).mean())
            one_9 = 0
            for  x in range(30,len(average_list)):
                if x < 900:
                    if np.std(one[x:x + 100] / one[x:].max()) < 2*one_5:
                        one_9 = average_list[x] / x #SI
                        break
                else:
                    if np.std(one[x:] / one[x:].max()) < 2*one_5:
                        one_9 = average_list[x] / x
                        break
            item.append('2_'+model)
            item.append(j)
            item.append(one_1)
            item.append(one_2)
            item.append(one_3)
            item.append(one_4)
            item.append(one_5)
            item.append(one_6)
            item.append(one_7)
            item.append(one_8)
            item.append(one_9)
            result_data.append(item)
            print('model:%s         ,seed:%s' % (model, j))
        model_name=['SCA_dqn','SC_ddqn','SCA_ddqn','SCA_double_drqn','SCA_gru','SCA_dueling_ddqn','qmix','vdn','vdn_noscale','iql','SCA_ddqn_noscale','cenddqn']
        for model_index,model in enumerate(['kmeans_dqn','kmeans_random_ddqn','kmeans_ddqn','kmeans_double_drqn','kmeans_gru','kmeans_dueling_ddqn','qmix','vdn','vdnnoscale','iql','kmeans_ddqnnoscale','cenddqn']):
            item = []
            try:
                one_m = pd.read_csv('result/%s_%s.csv' % (model, i))
            except:
                one_m = pd.read_csv('result/%s_%s.csv' % (model, i),encoding='gb18030')
            one_1 = np.mean(one_m['rewardAC'] + one_m['rewardBD'])  # AR
            one_2 = np.mean(np.sqrt(one_m['盾首水平偏差'] ** 2 + one_m['盾首垂直偏差'] ** 2))  # FAD
            one_3 = np.max(one_m['rewardAC'] + one_m['rewardBD'])  # Max
            one_4 = np.min(one_m['rewardAC'] + one_m['rewardBD'])  # Min
            one = one_m['rewardAC'] + one_m['rewardBD']
            one_5 = np.std(one[-500:]/one.max())  # Std
            one_6 = one_m[abs(one_m['rewardAC']) >= success_rate_2]['盾首水平偏差'].count()/10  # T1
            one_7 = one_m[abs(one_m['rewardBD']) >= success_rate_2]['盾首水平偏差'].count()/10  # T2
            one_8 = one_m[(abs(one_m['rewardAC']) >= success_rate_2) & (abs(one_m['rewardBD']) >= success_rate_2)][
                '盾首水平偏差'].count()/10  # TS
            list = one.to_list()
            average_list = []
            for index in range(len(list)):
                if index < 900:
                    li = list[index:index + 100]
                else:
                    li = list[index:]
                average_list.append(np.array(li).mean())
            one_9 = 0
            for  x in range(30,len(average_list)):
                if x < 900:
                    if np.std(one[x:x + 100] / one[x:].max()) < 2*one_5:
                        one_9 = average_list[x] / x# SI
                        break
                else:
                    if np.std(one[x:] / one[x:].max()) < 2*one_5:
                        one_9 = average_list[x] / x
                        break
            item.append(model_name[model_index])
            item.append(i)
            item.append(one_1)
            item.append(one_2)
            item.append(one_3)
            item.append(one_4)
            item.append(one_5)
            item.append(one_6)
            item.append(one_7)
            item.append(one_8)
            item.append(one_9)
            result_data.append(item)
            print('model:%s         ,seed:%s' % (model, i))
    result_data=pd.DataFrame(result_data,columns=['model','seed','AR','FAD','Max','Min',
                                                  'Std','T1','T2','TS','SI'])
    result_data.to_csv('result/train_total_result_data.csv',index=False)
    return 0

#Calculates the evaluation metric value of the testing set.
def get_test_metric(success_rate_2=400):
    result_data=[]
    for i in ['0','2','4']:
        model_name = ['SCA_dqn', 'SC_ddqn', 'SCA_ddqn', 'SCA_double_drqn', 'SCA_gru', 'SCA_dueling_ddqn',
                      'qmix', 'vdn', 'vdn_noscale', 'iql', 'SCA_ddqn_noscale', 'cenddqn']
        for model_index,model in enumerate(['kmeans_dqn','kmeans_random_ddqn','kmeans_ddqn','kmeans_double_drqn','kmeans_gru','kmeans_dueling_ddqn','qmix','vdn','vdnnoscale','iql','kmeans_ddqnnoscale','cenddqn']):
            item = []
            try:
                one_m = pd.read_csv('result/test/%s_%s.csv' % (model, i))
            except:
                one_m = pd.read_csv('result/test/%s_%s.csv' % (model, i),encoding='gb18030')
            one_1 = np.mean(one_m['rewardAC'] + one_m['rewardBD'])  # AR
            one_2 = np.mean(np.sqrt(one_m['盾首水平偏差'] ** 2 + one_m['盾首垂直偏差'] ** 2))  # FAD
            one_3 = np.max(one_m['rewardAC'] + one_m['rewardBD'])  # Max
            one_4 = np.min(one_m['rewardAC'] + one_m['rewardBD'])  # Min
            one = one_m['rewardAC'] + one_m['rewardBD']
            one_5 = np.std(one[-500:]/one.max())  # Std
            one_6 = one_m[abs(one_m['rewardAC']) >= success_rate_2]['盾首水平偏差'].count()  # T1
            one_7 = one_m[abs(one_m['rewardBD']) >= success_rate_2]['盾首水平偏差'].count()  # T2
            one_8 = one_m[(abs(one_m['rewardAC']) >= success_rate_2) & (abs(one_m['rewardBD']) >= success_rate_2)][
                '盾首水平偏差'].count()  # TS
            list = one.to_list()
            average_list = []
            for index in range(len(list)):
                if index < 990:
                    li = list[index:index + 10]
                else:
                    li = list[index:]
                average_list.append(np.array(li).mean())
            one_9 = 0
            for x in range(30,len(average_list)):
                if x < 900:
                    if np.std(one[x:x + 100] / one[x:].max()) < 2*one_5:
                        print(np.std(one[x:x + 100] / one[x:].max()))
                        one_9 = average_list[x] / x# SI
                        break
                else:
                    if np.std(one[x:] / one[x:].max()) < 2*one_5:
                        one_9 = average_list[x] / x
                        break
            item.append(model_name[model_index])
            item.append(i)
            item.append(one_1)
            item.append(one_2)
            item.append(one_3)
            item.append(one_4)
            item.append(one_5)
            item.append(one_6)
            item.append(one_7)
            item.append(one_8)
            item.append(one_9)
            result_data.append(item)
            print('model:%s         ,seed:%s' % (model, i))
    result_data=pd.DataFrame(result_data,columns=['model','seed','AR','FAD','Max','Min',
                                                  'Std','T1','T2','TS','SI'])
    result_data.to_csv('result/test_total_result_data.csv',index=False)
    return 0

#The evaluation metric values of randomized experiments are averaged, and Table 3 and Table 4 are obtained.
def get_mean_metric():
    train_data=pd.read_csv('result/train_total_result_data.csv')
    test_data = pd.read_csv('result/test_total_result_data.csv')
    group_train=train_data.groupby([train_data['model']]).mean()
    group_test = test_data.groupby([test_data['model']]).mean()
    group_train['seed']=group_train.index
    group_test['seed'] = group_test.index
    group_train.to_csv('result/train_total_result_data-mean.csv',index=False)
    group_test.to_csv('result/test_total_result_data-mean.csv', index=False)
    return 0

#Draw Fig. 4.
def get_figure_sdk(n=20):
    lables=['dqn','ddqn','double_drqn','dueling_ddqn']
    lable_text=['DQN','DDQN','Double DRQN','Dueling DDQN']
    x_lable=['(a)','(b)','(c)','(d)']
    plt.figure()
    for index,i in enumerate(lables):
        one_=[]
        doubel_=[]
        k_=[]
        lable=['1_%s'%lable_text[index],'2_%s'%lable_text[index],'SCA_%s'%lable_text[index]]
        for j in ['0','2','4']:
            try:
                one_m_=pd.read_csv('result/single/%s_%s.csv'%(i,j))
            except:
                one_m_ = pd.read_csv('result/single/%s_%s.csv' % (i,j),encoding='gb18030')
            one_.append(one_m_['rewardAC'].tolist())
            doubel_m_=pd.read_csv('result/double/%s_%s.csv'%(i,j))
            doubel_m_temp=doubel_m_['rewardAC']+doubel_m_['rewardBD']
            doubel_.append(doubel_m_temp.tolist())
        for k in ['0','2','4']:
            m=i
            k_m_=pd.read_csv('result/kmeans_%s_%s.csv'%(m,k))
            k_m_temp=k_m_['rewardAC']+k_m_['rewardBD']
            k_.append(k_m_temp.tolist())
        one_ = np.array(one_)
        doubel_ = np.array(doubel_)
        k_ = np.array(k_)
        one_=smooth(one_,n)
        doubel_=smooth(doubel_,n)
        k_=smooth(k_,n)
        data=[one_,doubel_,k_]
        df = []
        for i in range(len(data)):
            df.append(pd.DataFrame(data[i]).melt(var_name='episode', value_name='reward'))
            df[i]['algo'] = lable[i]
        df = pd.concat(df)
        df = df.reset_index()
        ax=plt.subplot(2,2,index+1)
        sns.lineplot(x="episode", y="reward", hue="algo", style="algo",linewidth=5, data=df,ax=ax,ci=40)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[0:], labels=labels[0:],loc='center right',fontsize=20)
        ax.set_xlabel('episode\n'+x_lable[index],fontweight='bold')
        ax.set_ylabel('average reward',fontweight='bold')
    plt.subplots_adjust(hspace=0.3,top=0.98)
    plt.show()
    return 0

#Draw Fig. 5.
def get_figure_km(n=20):
    lables = ['kmeans_dqn','kmeans_random_ddqn', 'kmeans_dueling_ddqn', 'kmeans_double_drqn', 'kmeans_ddqn','kmeans_gru','kmeans_ddqnnoscale','cenddqn','iql','vdn','qmix']
    plt.figure()
    lable = ['SCA_DQN', 'SC_DDQN','SCA_Dueling DDQN', 'SCA_Double DRQN', 'SCA_DDQN','SCA_GRU','SCA_DDQN_noscale','CenDDQN','IQL','VDN','QMIX']
    dqn = []
    r_ddqn=[]
    ddqn = []
    d_drqn = []
    d_ddqn = []
    gru = []
    ddqn_noscale = []
    cenddqn = []
    iql = []
    vdn = []
    qmix = []
    list_data=[dqn,r_ddqn,d_ddqn,d_drqn,ddqn,gru,ddqn_noscale,cenddqn,iql,vdn,qmix]
    for index,i in enumerate(lables):
        for k in ['0', '2', '4']:
            k_m_ = pd.read_csv('result/%s_%s.csv' % (i, k))
            k_m_temp = k_m_['rewardAC'] + k_m_['rewardBD']
            list_data[index].append(k_m_temp.tolist())
    data=[]
    for item in list_data:
        item=np.array(item)
        item=smooth(item,n)
        data.append(item)
    df = []
    for i in range(len(data)):
        df.append(pd.DataFrame(data[i]).melt(var_name='episode', value_name='reward'))
        df[i]['algo'] = lable[i]
    df = pd.concat(df)  # 合并
    df = df.reset_index()
    fig, ax = plt.subplots()
    sns.lineplot(x="episode", y="reward", hue="algo", style="algo",linewidth=5, data=df,ci=40)
    handles, labels = ax.get_legend_handles_labels()
    legend=ax.legend(handles=handles[0:], labels=labels[0:],bbox_to_anchor=(1.35,0.85),fontsize=20,handlelength=4)
    for line in legend.get_lines():
        line.set_linewidth(4.0)
    plt.xlabel('episode',fontweight='bold')
    plt.ylabel('average reward',fontweight='bold')
    # plt.subplots_adjust()
    plt.subplots_adjust(top=0.78,right=0.78)
    plt.show()
    return 0

#Successful count of the MADRL algorithm.
def get_success(data1,data2=False,test=False):
    success=[]
    if test:
        if data2==False:
            for index in range(len(data1)):
                i_sucess=len([i for i in data1[0:index+1] if i>400])
                success.append(i_sucess)
        else:
            for index in range(len(data1)):
                i_sucess=len([(i,j) for i,j in zip(data1[0:index+1],data2[0:index+1]) if i>400 and j>400])
                success.append(i_sucess)
    else:
        if data2==False:
            for index in range(len(data1)):
                i_sucess=len([i for i in data1[0:index+1] if i>400])/len(data1[0:index+1])
                success.append(i_sucess*100)
        else:
            for index in range(len(data1)):
                i_sucess=len([(i,j) for i,j in zip(data1[0:index+1],data2[0:index+1]) if i>400 and j>400])/len(data1[0:index+1])
                success.append(i_sucess*100)
    return success

#Successful count of non-MADRL algorithm.
def get_success1(data1,data2=False,test=False):
    success=[]
    if test:
        if data2==False:
            for index in range(len(data1)):
                i_sucess=len([i for i in data1[0:index+1] if i>400])
                success.append(i_sucess)
        else:
            for index in range(len(data1)):
                i_sucess=len([(i,j) for i,j in zip(data1[0:index+1],data2[0:index+1]) if i>400 and j>400])
                success.append(i_sucess)
    else:
        if data2==False:
            for index in range(len(data1)):
                i_sucess=len([i for i in data1[0:index+1] if i>600])/len(data1[0:index+1])
                success.append(i_sucess*100)
        else:
            for index in range(len(data1)):
                i_sucess=len([(i,j) for i,j in zip(data1[0:index+1],data2[0:index+1]) if i>300 and j>300])/len(data1[0:index+1])
                success.append(i_sucess*100)
    return success

#Draw Fig. 6.
def get_figure_meanSuccess():
    lables=['dqn','ddqn','double_drqn','dueling_ddqn']
    lable_text = ['DQN', 'DDQN', 'Double DRQN', 'Dueling DDQN']
    x_lable=['(a)','(b)','(c)','(d)']
    plt.figure()
    for index,i in enumerate(lables):
        one_=[]
        doubel_=[]
        k_=[]
        lable=['1_%s'%lable_text[index],'2_%s'%lable_text[index],'SCA_%s'%lable_text[index]]
        for j in ['0','2','4']:
            try:
                one_m_=pd.read_csv('result/single/%s_%s.csv'%(i,j))
            except:
                one_m_ = pd.read_csv('result/single/%s_%s.csv' % (i,j),encoding='gb18030')
            one_.append(get_success1(data1=one_m_['rewardAC'].tolist()))
            doubel_m_=pd.read_csv('result/double/%s_%s.csv'%(i,j))
            doubel_.append(get_success1(data1=doubel_m_['rewardAC'].tolist(),data2=doubel_m_['rewardBD'].tolist()))
        for k in ['0','2','4']:
            m=i
            k_m_=pd.read_csv('result/kmeans_%s_%s.csv'%(m,k))
            k_.append(get_success(data1=k_m_['rewardAC'].tolist(),data2=k_m_['rewardBD'].tolist()))
        data=[one_,doubel_,k_]
        df = []
        for i in range(len(data)):
            df.append(pd.DataFrame(data[i]).melt(var_name='episode', value_name='success'))
            df[i]['algo'] = lable[i]
        df = pd.concat(df)
        df = df.reset_index()
        ax=plt.subplot(2,2,index+1)
        sns.lineplot(x="episode", y="success", hue="algo", style="algo",linewidth=5,data=df,ax=ax,ci=0)
        handles, labels = ax.get_legend_handles_labels()
        legend=ax.legend(handles=handles[0:], labels=labels[0:],loc='center right',fontsize=20)
        ax.set_xlabel('episode\n'+x_lable[index],fontweight='bold')
        ax.set_ylabel('average success rate',fontweight='bold')
    plt.subplots_adjust(hspace=0.3, top=0.98)
    plt.show()
    return 0

#Draw Fig. 7.
def get_figure_kmMeanSuccess(test=False):
    lables = ['kmeans_dqn', 'kmeans_random_ddqn','kmeans_dueling_ddqn', 'kmeans_double_drqn', 'kmeans_ddqn','kmeans_gru','kmeans_ddqnnoscale','cenddqn','iql','vdn','qmix']
    plt.figure()
    lable = ['SCA_DQN', 'SC_DDQN', 'SCA_Dueling DDQN', 'SCA_Double DRQN', 'SCA_DDQN', 'SCA_GRU','SCA_DDQN_noscale','CenDDQN', 'IQL', 'VDN', 'QMIX']
    dqn = []
    r_ddqn=[]
    ddqn = []
    d_drqn = []
    d_ddqn = []
    gru = []
    ddqn_noscale = []
    cenddqn = []
    iql = []
    vdn = []
    qmix = []
    list_data=[dqn,r_ddqn,d_ddqn,d_drqn,ddqn,gru,ddqn_noscale,cenddqn,iql,vdn,qmix]
    if test:
        for index,i in enumerate(lables):
            for k in ['0', '2', '4']:
                try:
                    k_m_ = pd.read_csv('result/test/%s_%s.csv' % (i, k))
                except:
                    k_m_ = pd.read_csv('result/test/%s_%s.csv' % (i, k),encoding='gb18030')
                list_data[index].append(get_success(data1=k_m_['rewardAC'].tolist(),data2=k_m_['rewardBD'].tolist(),test=test))
    else:
        for index,i in enumerate(lables):
            for k in ['0', '2', '4']:
                k_m_ = pd.read_csv('result/%s_%s.csv' % (i, k))
                list_data[index].append(get_success(data1=k_m_['rewardAC'].tolist(),data2=k_m_['rewardBD'].tolist()))
    data=list_data
    df = []
    for i in range(len(data)):
        df.append(pd.DataFrame(data[i]).melt(var_name='episode', value_name='reward'))
        df[i]['algo'] = lable[i]
    df = pd.concat(df)
    df = df.reset_index()
    fig, ax = plt.subplots()
    sns.lineplot(x="episode", y="reward", hue="algo", style="algo",linewidth=5, data=df,ci=0)
    handles, labels = ax.get_legend_handles_labels()
    legend=ax.legend(handles=handles[0:], labels=labels[0:],framealpha=0.5,loc= 'center right',bbox_to_anchor=(1.35,0.5),fontsize=20,handlelength=4)
    for line in legend.get_lines():
        line.set_linewidth(4.0)
    plt.xlabel('episode',fontweight='bold')
    if test:
        plt.ylabel('average number of successes',fontweight='bold')
    else:
        plt.ylabel('average success rate',fontweight='bold')
    plt.subplots_adjust(top=0.78,right=0.78)
    plt.show()
    return 0



if __name__=='__main__':
    #Calculate the values in Tables 3 and 4.
    # get_train_metric()
    # get_test_metric()
    # get_mean_metric()


    # get_figure_sdk(n=20)#draw Fig. 4.
    # get_figure_km()#draw Fig. 5.
    get_figure_meanSuccess()#draw Fig. 6.
    get_figure_kmMeanSuccess()#draw Fig. 7.
    get_figure_kmMeanSuccess(test=True)#draw Fig. 8.

