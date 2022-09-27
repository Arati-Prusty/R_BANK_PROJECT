
bank_train=read.csv("D:\\R_PROJECT\\PROJECT5BANKDATA\\bank-full_train.csv")

bank_test=read.csv("D:\\R_PROJECT\\PROJECT5BANKDATA\\bank-full_test.csv")

library(dplyr)

glimpse(bank_train)

dim(bank_train)

head(bank_train)

glimpse(bank_test)

dim(bank_test)

head(bank_test)

#--------------------------------------------------------------------------

CreateDummies=function(data,var,freq_cutoff=0){
  t=table(data[,var])
  t=t[t>freq_cutoff]
  t=sort(t)
  categories=names(t)[-1]
  
  
  for(cat in categories){
    name=paste(var,cat,sep="_")
    name=gsub(" ","",name)
    name=gsub("-","_",name)
    name=gsub("\\?","Q",name)
    name=gsub("<","LT_",name)
    name=gsub("\\+","",name)
    name=gsub("\\/","_",name)
    name=gsub(">","GT_",name)
    name=gsub("=","EQ",name)
    name=gsub(",","",name)
    data[,name]=as.numeric(data[,var]==cat)
    
  }
  data[,var]=NULL
  return(data)
}

#---------------------------------------------------------------------

bank_test$y=NA

bank_train$data='train'

bank_test$data='test'

glimpse(bank_train)

glimpse(bank_test)


table(bank_train$y)

prop.table(table(bank_train$y))

bank_train$y=ifelse(bank_train$y =='yes',1,0)

table(bank_train$y)

glimpse(bank_train)

bank=rbind(bank_train,bank_test)

glimpse(bank)


#-----------------------------------------------------------------------------

lapply(bank,function(x) length(unique(x)))

names(bank)[sapply(bank, function(x) is.character(x))]


#----------------------------------------------------------------------

cat_cols=c('job','marital','education', 'default', 'housing', 'loan','contact',
           'month','poutcome')


for (cat in cat_cols){
  
  bank= CreateDummies(bank,cat,100)
}



#--------------------------------------------------------------------------



sum(sapply(bank,function(x) is.character(x)))

table(bank$y)

unique(bank$y)

prop.table(table(bank$y))


#-------------------------------------------------------------------------

sum(sapply(bank,function(x)sum(is.na(x))))

sapply(bank,function(x)sum(is.na(x)))

lapply(bank,function(x) sum(is.na(x)))


#----------------------------------------------------------------------


for(col in names(bank)){
  
  if(sum(is.na(bank[,col]>0 & !(col %in% c('data','y'))))){
    
    bank[is.na(bank[,col]),col] = mean(bank[bank$data=='train',col],na.rm=T)
  }
}

lapply(bank,function(x) sum(is.na(x)))  


#-----------------------------------------------------------------------

bank_train=bank %>%  
  
  filter(data =='train') %>%
  
  select(-data)


bank_test=bank %>%  
  
  filter(data =='test') %>%
  
  select(-data,-y)

glimpse(bank_train)

dim(bank_train)

glimpse(bank_test)

dim(bank_test)

#-----------------------------------------------------------------------------

any(is.na(bank_train))

any(is.na(bank_test))

#----------------------------------------------------------------------------

set.seed(2)

s=sample(1:nrow(bank_train),0.8*nrow(bank_train))

bank_train1=bank_train[s,]

bank_train2=bank_train[-s,]

nrow(bank_train1)

nrow(bank_train2)

nrow(bank_test)


#---------------------------------------------------------------------------

# for gbm no need to convert target in to factor like random forest

library(cvTools)

library(randomForest)

library(tree)

library(ggplot2)

library(gbm)

library(lattice)

library(robustbase)

#--------------------------------------------------------------------------

table(bank_train$y)

head(bank_train1)

head(bank_train2)

dim(bank_train1)

dim(bank_train2)

dim(bank_test)


#-----------------------------------------------------------------------------

params =list( interaction.depth=c(5,10),n.tree=c(50,100,500),
              
              shrinkage=c(0.1,0.01,0.001),n.minobsinnode=c(2,5))

expand.grid(params)


param =list(interaction.depth = c(5,10,15,20),
            
            n.tree= c(50,100,200,500),
            
            shrinkage=c(0.1,0.01,0.001),
            
            n.minobsinnode=c(2,5,10))

size_grid=expand.grid(param)

head(size_grid)


#--------------------------------------------------------------------------


subset_paras=function(full_list_para,n=10)
{
  all_comb=expand.grid(full_list_para)
  
  s=sample(1:nrow(all_comb),n)
  
  subset_para=all_comb[s,]
  
  return(subset_para)
  
}

#---------------------------------------------------------------------------



mycost_auc = function(y,yhat)
  
{
  roccurve=pROC::roc(y,yhat)
  
  score=pROC::auc(roccurve)
  
  return(score)
}

#----------------------------------------------------------------------------

num_trials= 50

my_params=subset_paras(param,num_trials)

dim(my_params)

#-----------------------------------------------------------------------------

myauc=0

for (i in 1:num_trials)
  
{
  print(paste('starting iteration:',i))
  
  params=my_params[i,]
  
  k=cvTuning(gbm, y~., -ID,
             
             data=bank_train1,
             
             tuning = params,
             
             args=list(distribution="bernoulli"),
             
             folds=cvFolds(nrow(bank_train1), K=10,type="random"),
             
             cost=mycost_auc, seed=2,
             
             predictArgs = list(type="response", n.tree=params$n.trees))
  
  
  score.this=k$cv[,2]
  
  print(score.this)
  
  if(score.this>myauc){
    
    myauc=score.this
    
    best_params=params
    
    
  }
  
  
}


print(best_params)

myauc

#-----------------------------------------------------------------------------

myauc=0.9443646

best_params= data.frame(interaction.depth=10, n.tree=200,
                        shrinkage=0.1,n.minobsinnode=5)





best_params



bank.gbm.final=gbm(y~.,data=bank_train1,interaction.depth = best_params$interaction.depth,
                   n.trees =best_params$n.tree,shrinkage = best_params$shrinkage,
                   n.minobsinnode = best_params$n.minobsinnode,distribution = "bernoulli" )




#-------------------------------------------------------------------------------

bank_train1.score=predict(bank.gbm.final,newdata = bank_train1,type="response")

bank_train1.score

max(bank_train1.score)

min(bank_train1.score)


bank_train2.score=predict(bank.gbm.final,newdata = bank_train2,type="response")

max(bank_train2.score)
  
min(bank_train2.score)

#--------------------------------------------------------------------------------

real=bank_train1$y

bank_test.score=predict(bank.gbm.final,newdata = bank_test,type="response")

length(bank_test.score)




#----------------------------------------------------------------------------

real=bank_train2$y


rocit=ROCit::rocit(score = bank_train2.score,class=real)

kplot=ROCit::ksplot(rocit,legend=F)

kplot

KS = 0.77

cutoff= 0.084


#---------------------------------------------------------------------------

# test.predicted always slight more than cutoff

#unique(test.predicted)

#length(test.predicted)

test.predicted=ifelse(test.predicted > 0.09,"Yes","No")

table(test.predicted)

prop.table(table(test.predicted))

table(bank_train$y)

prop.table(table(bank_train$y))

write.csv(test.predicted, "ARATI_PRUSTY_P5_part2.csv",row.names=F)



#-----------------------------------------------------------------------------


