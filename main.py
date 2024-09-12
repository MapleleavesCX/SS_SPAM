
from ss_spam import pp_spam_filter

# 示例使用
if __name__ == '__main__':

    # 文本地址
    spam1_path = './data/spam1.csv'
    column_X_spam1='Message'
    column_y_spam1='Category'
    classif_spam1={'ham':0, 'spam':1}
    encoding_spam1='ctf-8'

    spam2_path = './data/spam2.csv'
    classif_spam2={}
    column_X_spam2='text'
    column_y_spam2='spam'
    encoding_spam2='ISO-8859-1'
    
    # 这里由于 spam2.csv 中的标签列已经标准化，故直接返回，传入 {} 即可

    # 模型选择参数
    model_id = 1001
    # 成员设置
    parties = ['alice', 'bob']
    # 成员数据集划分： alice / bob 划分给每个节点1/2的 X 部分(默认按列)
    X_par = {
        'alice':(0,0.5),
        'bob':(0.5,1)
    }
    Y_par = {
        'alice':(0,1)
    }
    # ********************************************初始化***********************************************
    print('初始化隐私保护垃圾邮件过滤器...')
    run = pp_spam_filter(
        model_id, parties=parties, extract_feat_method='tfidf', reduce_dim_method='pca')
    # *******************************************文本处理**********************************************
    print('文本处理...')
    run.Textprocessor(_input_file_path=spam1_path, 
                      message_column_name='Message', labels_column_name='Category', 
                      classification=classif_spam1)

    # ******************************************数据集划分*********************************************
    print('划分数据...')
    run.data_divider(direction='c', X_partitions=X_par, y_partitions=Y_par)

    # *******************************************训练*************************************************
    print('训练模型...')
    run.train()

    # *******************************************评估*************************************************
    print('评估模型...')
    run.evaluation('alice')

    # *******************************************输出*************************************************
    print('过滤器测试：')
    run.filter(memberKey='alice', input_text_path=spam1_path, 
            message_column_name='Message', labels_column_name='Category', classify=classif_spam1)
