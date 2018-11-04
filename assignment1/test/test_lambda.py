'''
	这个例子模拟了q3的过程，主要是lambda传来串去，参数搞晕了。
	核心是x0，从sgd的x0，
		lambda vec: word2vec_sgd_wrapper(skipgram, vec <----看这个vec
		其实就是f(x0)<====就是这个x0
		而这个x0，又是sg的第2个参数'x0'（第一个参数是lambda）

	不难，但是绕了一圈，主要是不停第可以把参数向深层的调用中，传递，我理解
	这样就不难理解q3_run.py 36行的vec，
	就是未来谁来调用这个word2vec_sgd_wrapper时候传入的参数就是这个vec
		wordVectors = sgd(
		    lambda vec: word2vec_sgd_wrapper(
		        skipgram, #skipgram算法的函数指针
		        tokens, #词表
		        vec, #每行的词向量？还是每一列？干嘛？？？
		        dataset, 
		        C,
		        negSamplingCostAndGradient),#负采样的时候的梯度？
		    wordVectors, 	那，是谁调用并传入了这个vec呢？是q3_sgd.py的sgd函数，88行
		cost, grad = f(x)
		表面上只有x一个参数，但其实还有一堆的其他参数，也就是上面几行描述的tokens...那些参数
		x是谁呢？就是上面的第二个参数“wordVectors”
	唉，lambda其实是一个函数编程，这样玩，只是为了让核心参数wordVectors晚点绑定到函数里而已。
'''

def sgd(f, x0, step):
	print("sgd: f %s, x0 %s, step %s" % (str(f),str(x0), str(step)) )
	print("sgd: call f(x0)")
	f(x0)

def skipgram(vec,p1,p2):
	print("skipgram: vec %s, p1 %s, p2 %s" % (str(vec),str(p1),str(p2)) )

def word2vec_sgd_wrapper(word2vecModel, vec, p1, p2):
	print("word2vec_sgd_wrapper: word2vecModel, %s,vec %s,p1 %s, p2 %s" % 
		(str(word2vecModel),str(vec),str(p1),str(p2)) )
	print("word2vec_sgd_wrapper: call wordVecModel(vec,p1,p2)")
	word2vecModel(vec,p1,p2)

wordVectors = sgd(
    lambda vec: word2vec_sgd_wrapper(skipgram, vec, ['a','b','c'], 123),
    	'x0',
    	123)


