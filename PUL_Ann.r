library(nnet)
library(AUC)
library(splines)
library(mgcv)
library(caret)
library(lattice)
library(nlme)
library(ggplot2)

hd<-"E:\\PUL-MPM\\Data"
outdir<-paste(hd,"Result\\PUL_Ann\\",sep="")
codedir<-paste(hd,"Code\\PUL_Ann\\",sep="")
variable_num <- 5
for (i in 1:10)
{
    source(paste(codedir,"calcError.r", sep = ""))
    source(paste(codedir,"best_t.r", sep = ""))
    set.seed(1);
    data <- read.csv(paste(hd,"allpixel.csv",sep=""),header=TRUE);
	test_path <- paste(hd,'Sample\\test',as.character(i),'.csv', sep = "");
	test <- read.csv(test_path,header = TRUE);    
	train_path <- paste(hd,'Sample\\tra',as.character(i),'.csv', sep = "");
	tra <- read.csv(train_path,header = TRUE);
	validate_path <- paste(hd,'Sample\\val',as.character(i),'.csv', sep = "");
	val <- read.csv(validate_path,header = TRUE);
	model <- nnet(x = tra[,2:variable_num+1], y = tra[,1], size = 4, entropy = TRUE, decay = 0.1, maxit = 800)					 
	# calculate c  and threshold and threshold
	positive <- subset(val, label == 1)	
	ymean_c <- predict(model, positive[,2:variable_num+1], type = "raw")	
	c <- mean(ymean_c);	
	c_file <- paste(outdir,'c\\','PUL_Ann_c',as.character(i),'.csv',sep="")
	write("c", file = c_file, append = FALSE, sep = " ");
	write.table(c, file = c_file, append = TRUE, sep = ",", row.names = FALSE, col.names = FALSE)	
	yfit_val <- predict(model, val[,2:variable_num+1], type = "raw")	
	yfit_val_post <- yfit_val/c;
	yfit_val_post[yfit_val_post<0]<-0;
	yfit_val_post[yfit_val_post>1]<-1;
	mse <- mean((yfit_val - val[, 1])^2);											# mean-of-squares error function
	mse_file <- paste(outdir,'MSE\\','PUL_Ann_MSE',as.character(i),'.csv',sep="");
	write("mse", file = mse_file, append = FALSE, sep = " ");
	write.table(mse, file = mse_file, append = TRUE, sep = ",", row.names = FALSE, col.names = FALSE)
	# hthr
	hthr0<-best_t(yfit_val, val[,1]); # hthr0	
	hthr<-best_t(yfit_val_post, val[,1]);  # hthr
	hthr_file <- paste(outdir,'hthr\\','PUL_Ann_hthr',as.character(i),'.csv',sep="")
	write("hthr", file = hthr_file, append = FALSE, sep = " ");
	write.table(hthr, file = hthr_file, append = TRUE, sep = ",", row.names = FALSE, col.names = FALSE)		
	# acc
	test_predict_pre <- predict(model, test[,2:variable_num+1], type = "raw")	
	test_cls_pre<-(test_predict_pre>=hthr0);
	test_cls_pre[test_cls_pre==FALSE]=0;
	test_cls_pre[test_cls_pre==TRUE]=1;
	acc_pre <- calcError(test[,1],test_cls_pre,test_predict_pre,c(1,2)); 
	acc_pre_file <- paste(outdir,'pre_precision\\','PUL_Ann_acc',as.character(i),'.csv',sep="")
	write("Fpb,Auc,R", file = acc_pre_file, append = FALSE, sep = " ");
	write.table(acc_pre, file = acc_pre_file, append = TRUE, sep = ",", row.names = FALSE, col.names = FALSE)	
	#  post acc
	test_predict_post0 <- test_predict_pre/c;
	test_predict_post[test_predict_post<0]<-0;
	test_predict_post[test_predict_post>1]<-1;
	test_cls_post<-(test_predict_post>=hthr)	
	test_cls_post[test_cls_post==FALSE]=0;
	test_cls_post[test_cls_post==TRUE]=1;
	acc_post <- calcError(test[,1],test_cls_post,test_predict_post0,c(1,2));  # precision
	acc_post_file <- paste(outdir,'precision\\','PUL_Ann_acc',as.character(i),'.csv',sep="")
	write("Fpb,Auc,R", file = acc_post_file, append = FALSE, sep = " ");
	write.table(acc_post, file = acc_post_file, append = TRUE, sep = ",", row.names = FALSE, col.names = FALSE)	
	# for ensemble
	test_Yfit_file <- paste(outdir,'TEST_pred\\','PUL_Ann_test_yfit',as.character(i),'.csv',sep="");
	write("label", file = test_Yfit_file, append = FALSE, sep = " ");
	write.table(test_predict_post, file = test_Yfit_file, append = TRUE, sep = ",", row.names = FALSE, col.names = FALSE)	
	# predict allpixel
	predict_pre <- predict(model, data[,2:variable_num+1], type = "raw")	
	predict_post <- predict_pre/c;   
	predict_post[predict_post<0]<-0;
	predict_post[predict_post>1]<-1;
	class_post<-predict_post>=hthr
	class_post[class_post==FALSE]=0;
	class_post[class_post==TRUE]=1;	
	yfit_file_post <- paste(outdir,'predict\\','PUL_Ann_yfit',as.character(i),'.csv',sep="");
	write("label", file = yfit_file_post, append = FALSE, sep = " ");
	write.table(predict_post, file = yfit_file_post, append = TRUE, sep = ",", row.names = FALSE, col.names = FALSE)	
	class_file <- paste(outdir,'class\\','PUL_Ann_cls',as.character(i),'.csv',sep="");
	write("label", file = class_file, append = FALSE, sep = " ");
	write.table(class_post, file = class_file, append = TRUE, sep = ",", row.names = FALSE, col.names = FALSE)
}

 
