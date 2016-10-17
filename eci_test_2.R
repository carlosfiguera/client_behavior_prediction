    #27 de mayo de 2016
#authors: Óscar Barquero, Carlos Figuera, Rebeca Goya

# Read data into dataframe #####
setwd('~/Dropbox/project_data/')
eci = read.table("../data_set_3/data_logic-data_logic-logic-2016-05-0107.csv", header=T,sep = ',')
# Preliminar checks: summary, dimensions, names, edit
names(eci)
summary(eci)
dim(eci)
#edit(eci)
summary(eci)

#Con el summary verificamos que las siguientes vairables tiene nan:

#
# sid_avgduration: duración media de las sesiones (¿qué significa NaN?), -> asumimos que cuando NaN, esto significa 0 en los últimos 30 días
# count_trans: num de transacciones totales por navegador (¿qué significa NaN?) -> asumimos que cuando NaN, ésto significa 0 en los últimos 30 días
# bounces: número de veces que rebota con la página (¿qué es NaN?) -> cuanod NaN asumimos 0
# pageviews: total de páginas vistas del usuario (¿qué es NaN?) -> cuando es NaN asumimos 0
# pageviews_persid: media de páginas vista por sesión del usuario (¿qué es NaN?) -> cuando es NaN asumimos 0
# timeOnSite: tiempo en el site (¿qué es NaN?) -> asumismos que NaN quiere decir que en los 30 días no ha visitado site.
# totalTransactionRevenue: ingresos del usuario -> sería una y2 para el caso de multclasificación o regresión.

#Reacondicionamos el dataframe de acuerdo a las consideraciones anteriores
# el resultado es asignar a todos los na dentro del dataframe el valor 0
eci[is.na(eci)] = 0

#Nota en trasaction_predweek
summary(eci$trasaction_predweek)
#Aparentemente esta variable debería ser booleana, de forma que 1 implica transacción, 0 no transacción.
#pero en nuestros datos esta variable aparece con los siguientes valores 0,1,2,3 y 4.
#Vamos a crear una nueva variable transaction: 1 cuando ha habido transacción (trasaction_predweek >= 1), 0 cuando no ha habido transacción (trasaction_predweek = 0) 
#eci$transaction = eci$trasaction_predweek
#eci$transaction[eci$transaction > 1] = 1

eci$transaction = as.factor(eci$trasaction_predweek)

#en esta versión de los datos trasaction_predweek ya está convertida a 0 1
#attach(eci)
summary(eci)
plot(eci$count_trans)
plot(eci$count_trans,eci$trasaction_predweek)


#remove some maximun numbers which in this data base seemede to be enormour according to the distribution
boxplot(eci$items_cart)
boxplot(eci$p_funnel)

#------------------------------------------------
# Análisis con arboles de decisión binarios
#------------------------------------------------
library(tree)

#vamos a crear un data frame que sin las variables trasaction_predweek ni totalTransactionReveneu

#eci_to_tree = subset(eci,select = -c(totalTransactionRevenue,trasaction_predweek,fullVisitorId,count_trans,items_cart,p_funnel))
#eci_to_tree = subset(eci,select = -c(totalTransactionRevenue,trasaction_predweek,fullVisitorId,p_funnel,count_trans,items_cart,e_add_cart))
eci_to_tree = subset(eci,select = -c(totalTransactionRevenue,trasaction_predweek,fullVisitorId,ltsid_contenido,e_item_row))
#attach(eci_to_tree)

#variables booleanas con as.factor

labs = names(eci_to_tree)
for (i in 1:length(eci_to_tree)){
  if(is.factor(eci_to_tree[,i])){
    t = table(eci_to_tree[,i],eci_to_tree$transaction)
    cat(c(labs[i],'transaction\n'))
    cat(c(t[1],'\t',t[2],'\n'))
    cat(c(t[3],'\t',t[4],'\n'))
    cat("--------------------------------\n\n")
  }
  else{
    boxplot(eci_to_tree[,i]~eci_to_tree$transaction,xlab = 'transaction',ylab = labs[i])
  }
}

#---------------
#some Exploratory analysis
#-----------------------------

#datos muy desbalanceados, vamos a quedarnos con el mismo número de datos de 0 que de 1.
id_transaction_0 = sample(which(eci_to_tree$transaction==0),sum(eci_to_tree$transaction == 1)+100)
#create new data frame with balanced data
balanced_eci = eci_to_tree[c(id_transaction_0,which(eci_to_tree$transaction==1)),]


summary(balanced_eci)

#remove variables with 0 values in one of the levels

balanced_eci = subset(balanced_eci,select = -c(fsid_contenido,lsid_contenido,ltsid_acuerdo,ltsid_social))
summary(balanced_eci)

#exploratory analysis
#pairs(balanced_eci)
data_frame_for_corr = eci[c(id_transaction_0,which(eci_to_tree$transaction==1)),]
data_frame_for_corr$transaction = as.numeric(as.character(data_frame_for_corr$transaction))
data_frame_for_corr$newvisit = as.numeric(data_frame_for_corr$newvisit)
cor(data_frame_for_corr)
#pairs(data_frame_for_corr)

#look for features correlated with transaction
labs = names(balanced_eci)
#boxplot and ...
for (i in 1:length(balanced_eci)){
  if(is.factor(balanced_eci[,i])){
    t = table(balanced_eci[,i],balanced_eci$transaction)
    cat(c(labs[i],'transaction\n'))
    cat(c(t[1],'\t',t[2],'\n'))
    cat(c(t[3],'\t',t[4],'\n'))
    cat("--------------------------------\n\n")
  }
  else{
    
    if (sd(balanced_eci[,i]) < 0.00001){
      cat("Standard deviatoin is zero")
    }
    else{
      #correlation_with_tr = cor(balanced_eci[,i],balanced_eci$transaction_predweek)
      correlation_with_tr_pred = cor(balanced_eci[,i],as.numeric(balanced_eci$transaction))
      cat("Correlation ",labs[i],' vs. Transaction\n')
      #cat("rho = ",correlation_with_tr,'\n')
      #cat("Correlation ",labs[i],' vs. Transaction predweek\n')
      cat("rho = ",correlation_with_tr_pred,'\n')
      cat('------------------------\n')
      boxplot(balanced_eci[,i]~balanced_eci$transaction,xlab = 'transaction',ylab = labs[i])
    }
  }
}

#--------------#--------------#--------------#--------------
#
# Split data in training and test
#
#--------------#--------------#--------------#--------------

# 80 % training
# 20 test

## 75% of the sample size
smp_size = floor(0.80 * nrow(balanced_eci))

## set the seed to make your partition reproductible
set.seed(123)
train_ind = sample(1:nrow(balanced_eci), size = smp_size)


train_eci = balanced_eci[train_ind,]
test_eci = balanced_eci[-train_ind,]

#scale numerica variables
num_id = sapply(train_eci,is.numeric)
fact_id = sapply(train_eci,is.factor)
scaled_train = scale(train_eci[,num_id])
scaled_train_eci = data.frame(scaled_train)
scaled_train_eci = cbind(scaled_train_eci,train_eci[,fact_id])
scaled_train_eci$transaction = train_eci$transaction
scaled_train_eci[is.na(scaled_train_eci)] = 0

#scaling test set using the data (mean and std) computed on training
scaled_test = scale(test_eci[,num_id],center = attr(scaled_train,'scaled:center'), scale = attr(scaled_train,'scaled:scale'))
scaled_test_eci = data.frame(scaled_test)
scaled_test_eci = cbind(scaled_test_eci,test_eci[,fact_id])
scaled_test_eci$transaction = test_eci$transaction
scaled_test_eci[is.na(scaled_test_eci)] = 0


#############################################################################
#
# tree simple analysis
#
##############################################################################
library(tree)
tree.eci = tree(train_eci$transaction~.,train_eci)
#tree.eci = tree(eci_to_tree$transaction~.,eci_to_tree)

summary(tree.eci)
plot(tree.eci)
text(tree.eci,pretty = 0)

pred = predict(tree.eci,test_eci,type="class")
table(pred,test_eci$transaction)
t=table(pred,test_eci$transaction)

accuracy_tree = (t[1] + t[4]) / nrow(test_eci)
accuracy_tree
#some pruning
set.seed(3)
cv.tree_eci = cv.tree(tree.eci, FUN=prune.misclass)

#vamos a probar con 5 que tiene la misma desvianza que con 8
prune.tree_eci = prune.misclass(tree.eci,best = 5)
plot(prune.tree_eci)
text(prune.tree_eci,pretty= 0 )

#pred_2 = predict(prune.tree_eci,scaled_test_eci,type="class")
#table(pred_2,test_eci$transaction)
#t=table(pred_2,test_eci$transaction)

#accuracy = (t[1] + t[4]) / nrow(test_eci)
#accuracy
#----------------------------
# Logistic regression
#----------------------------



#En este punto vamos a pasar a realizar el análisis mediante regresión logística
library(glm)
#eci.lgr(balanced_eci$)
train_eci_na = na.omit(train_eci)
eci.lgr = glm(scaled_train_eci$transaction~.,data = scaled_train_eci,family = binomial)

summary(eci.lgr)

y_pred = predict(eci.lgr,scaled_test_eci,type = 'response')

colors = c(rep("royalblue",1),rep("orange",1))
par(family = 'serif',font.axis = 12,font.lab = 12)
boxplot(y_pred~ test_eci$transaction,ylab = 'Predictec Probability of Transaction',col = colors,names = c("Transaction = 0","Transaction = 1"))
#y_pred are the probabilities of transactions
#convert to 0 and 1
y_pred[y_pred>0.5] = 1
y_pred[y_pred <= 0.5] = 0

table(y_pred,test_eci$transaction)
t=table(y_pred,test_eci$transaction)

accuracy = (t[1] + t[4]) / nrow(test_eci)
accuracy


#-------------------##-------------------#
#
# Ridge logistic regression and L1 penalized logistic regression
#
##-------------------##-------------------#

library(glmnet)
# First convert to data matrix
x = model.matrix(scaled_train_eci$transaction~.,scaled_train_eci)[,-1]
y = scaled_train_eci$transaction

grid = 10^seq(10,-4,length=100)
eci_ridge.mod = glmnet(x,y,alpha = 0,lambda = grid,family = 'binomial')
#dim(coef(eci_ridge.mod))


#choose lambda using CV
set.seed(1)
cv.out = cv.glmnet(x,y,alpha = 0,family = 'binomial')
plot(cv.out)
best_lambda = cv.out$lambda.1se

x_test = model.matrix(scaled_test_eci$transaction~.,scaled_test_eci)[,-1]
y_test = scaled_test_eci$transaction
ridge_pred = predict(eci_ridge.mod,s = best_lambda,newx = x_test,type = 'response')

ridge_pred[ridge_pred>0.5] = 1
ridge_pred[ridge_pred <= 0.5] = 0
table(ridge_pred,test_eci$transaction)
t=table(ridge_pred,test_eci$transaction)

accuracy = (t[1] + t[4]) / nrow(test_eci)
accuracy

#Now L1 penalized 

lasso.mod = glmnet(x,y,alpha = 1,lambda = grid,family = 'binomial')
plot(lasso.mod,label = TRUE)
labs = names(scaled_train_eci)
#Nice plot
par(mar=c(4.5,4.5,1,4))
plot(lasso.mod)
vnat=coef(lasso.mod)
vnat=vnat[-1,ncol(vnat)] # remove the intercept, and get the coefficients at the end of the path
axis(4, at=vnat,line=-.5,label=labs[1:47],las=1,tick=FALSE, cex.axis=0.8) 

set.seed(1)
cv.out_lasso= cv.glmnet(x,y,alpha=1,family = 'binomial')
best_lambd_lasso = cv.out$lambda.min
l1_pred = predict(lasso.mod,s = best_lambd_lasso,newx = x_test,type = 'response')
plot(cv.out_lasso)

colors = c(rep("royalblue",1),rep("orange",1))
par(family = 'serif',font.axis = 12,font.lab = 12)
boxplot(l1_pred~ test_eci$transaction,ylab = 'Predicted Probability of Transaction (L1 norm)',col = colors,names = c("Transaction = 0","Transaction = 1"))

l1_pred[l1_pred>0.5] = 1
l1_pred[l1_pred <= 0.5] = 0
table(l1_pred,test_eci$transaction)
t=table(l1_pred,test_eci$transaction)
accuracy = (t[1] + t[4]) / nrow(test_eci)
accuracy


out = glmnet(x,y,alpha = 1,lambda=grid,family = 'binomial') #aquí habría que usar toda la base de datos, train y test
lasso.coef = predict(out,type = 'coefficients',s=best_lambd_lasso)
lasso.coef
lasso.coef[lasso.coef!=0]
sort(abs(lasso.coef[lasso.coef!=0]),decreasing = TRUE)

c<-coef(out,s=best_lambd_lasso,exact=TRUE)
inds<-which(c!=0)
variables<-row.names(c)[inds]
variables<-variables[2:length(variables)]
cat(variables,'\n')
cat(round(c[inds[2:length(inds)]],2),'\n')


coef2 <- function(fit, s){
  cf <- as.matrix(coef(fit, s=s))
  cf <- data.frame(coef=cf[cf[,1] != 0 , ])
  cf$vars <- row.names(cf)
  cf[order(abs(cf$coef), decreasing=T), ]
}

coef2(out,s = best_lambd_lasso)


###############################
# Saving model and scaling transformations
###############################

save(scaled_train,file = 'scaling_object.Rdata')
save(eci.lgr,file = 'eci_lgr.rda')


##############################
# Load transformation and eci logistic regresion model
#############################

load(file = "eci_lgr.rda")
load(file = 'scaling_object.Rdata')
