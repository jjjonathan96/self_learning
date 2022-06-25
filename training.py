'''

All are result images images
'''

master = [0,1,2]
teacher = [0,1,2,3,4,5,6,7,8]
student = [0,1,2,3,4,5,6,7,8,9,10]


for i in master:
    '''
    master equation
    '''

    for j in teacher:
        '''
        ResNet50
        '''

        for k in student:
            '''
            ViT s/8
            
            '''
