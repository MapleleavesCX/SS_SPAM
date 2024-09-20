from ss_spam import pp_spam_filter
from ml.toolfunc import TerminalColors

# 示例使用
if __name__ == '__main__':

    # 数据
    spam1_path = './data/spam1.csv'
    column_X_spam1='Message'
    column_y_spam1='Category'
    classif_spam1={'ham':0, 'spam':1}
    encoding_spam1='ctf-8'

    spam2_path = './data/spam2.csv'
    classif_spam2={0:0, 1:1}
    column_X_spam2='Body'
    column_y_spam2='Label'
    encoding_spam2='ctf-8'

    # 集群仿真参数 local 则表示本地单节点仿真
    ray_addr = '192.168.211.129:9000'

    # 集群仿真参数：需要修改IP
    alice_ip = '192.168.211.129'
    alice_port = '9100'

    bob_ip = '192.168.211.129'
    bob_port = '9200'

    carol_ip = '192.168.211.129'
    carol_port = '9300'

    # 模型选择参数
    model_id = 2001

    # 成员设置
    parties = ['alice', 'bob', 'carol']

    # 集群设置
    cluster_def={
        'nodes': [
            {
                'party': 'alice',
                'address': f'{alice_ip}:{alice_port}',
                'listen_addr': f'0.0.0.0:{alice_port}'
            },
            {
                'party': 'bob',
                'address': f'{bob_ip}:{bob_port}',
                'listen_addr': f'0.0.0.0:{bob_port}'
            },
            {
                'party': 'carol',
                'address': f'{carol_ip}:{carol_port}',
                'listen_addr': f'0.0.0.0:{carol_port}'
            },
        ],
    }


    # 成员数据集划分设置：
    X_par = {
        'carol':(0,0.5),
        'bob':(0.5,1)
    }
    Y_par = {
        'carol':(0,0.5),
        'bob':(0.5,1)
    }
    
    # 训练时参数（仅NN模型）
    params_nn = {
        'epochs':5,
        'sampler_method':"batch",
        'batch_size':32,
        'aggregate_freq':1,
    }

    # ********************************************初始化***********************************************
    print(TerminalColors.BRIGHT_BLUE + '初始化隐私保护垃圾邮件过滤器...' + TerminalColors.END)
    run = pp_spam_filter(
        model_id, parties=parties, extract_feat_method='tfidf', reduce_dim_method='pca',
        ray_addr=ray_addr, cluster_def=cluster_def
        )
    # *******************************************文本处理**********************************************
    print(TerminalColors.BRIGHT_BLUE + '文本处理...' + TerminalColors.END)
    run.Textprocessor(_input_file_path=spam1_path, 
                      message_column_name=column_X_spam1, 
                      labels_column_name=column_y_spam1, 
                      classification=classif_spam1)
    print(TerminalColors.BRIGHT_BLUE + '处理完成！' + TerminalColors.END)

    # ******************************************数据集划分*********************************************
    print(TerminalColors.BRIGHT_BLUE + '划分数据...' + TerminalColors.END)
    run.data_divider(direction='r', X_partitions=X_par, y_partitions=Y_par)

    # *******************************************训练*************************************************
    print(TerminalColors.BRIGHT_BLUE + '训练模型...' + TerminalColors.END)
    run.train(params=params_nn)
    print(TerminalColors.BRIGHT_GREEN + '训练完成！' + TerminalColors.END)

    # *******************************************评估*************************************************
    print(TerminalColors.BRIGHT_BLUE + '评估模型...' + TerminalColors.END)
    run.evaluation('bob')
    print(TerminalColors.BRIGHT_GREEN + '评估结束！' + TerminalColors.END)

    # *******************************************输出*************************************************
    print(TerminalColors.BRIGHT_BLUE + '过滤器测试：' + TerminalColors.END)
    while True:
        chose = input('采用哪个数据集测试？[spam1/spam2]')
        if chose == 'spam1':
            run.filter(input_text_path=spam1_path, 
                    message_column_name=column_X_spam1, 
                    labels_column_name=column_y_spam1, 
                    classify=classif_spam1)
        elif chose == 'spam2':
            run.filter(input_text_path=spam2_path, 
                    message_column_name=column_X_spam2, 
                    labels_column_name=column_y_spam2, 
                    classify=classif_spam2)
        else:
            print('错误！不存在这样的数据集')
        
        chose2 = input('是否继续？[y/n]')
        if chose2 == 'n':
            break

