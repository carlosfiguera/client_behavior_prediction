#23 de mayo de 2016
#authors: Óscar Barquero, Carlos Figuera, Rebeca Goya

# Read data into dataframe #####
setwd('~/Dropbox/Cocktail/cocktail_project/')
eci = read.table("../data_set_2/data_logic-05010508.csv", header=T,sep = ',')
# Preliminar checks: summary, dimensions, names, edit
names(eci)
summary(eci)
dim(eci)
#edit(eci)


#Con el summary verificamos que las siguientes vairables tiene nan:

#
# sid_avgduration: duración media de las sesiones (¿qué significa NaN?), -> asumimos que cuando NaN, esto significa 0 en los últimos 30 días
# count_trans: num de transacciones totales por navegador (¿qué significa NaN?) -> asumimos que cuando NaN, ésto significa 0 en los últimos 30 días
# bounces: número de veces que rebota con la página (¿qué es NaN?) -> cuanod NaN asumimos 0
# pageviews: total de páginas vistas del usuario (¿qué es NaN?) -> cuando es NaN asumimos 0
# pageviews_persid: media de páginas vista por sesión del usuario (¿qué es NaN?) -> cuando es NaN asumimos 0
# timeOnSite: tiempo en el site (¿qué es NaN?) -> asumismos que NaN quiere decir que en los 30 días no ha visitado site ECI.
# totalTransactionRevenue: ingresos del usuario -> sería una y2 para el caso de multclasificación o regresión.

#Reacondicionamos el dataframe de acuerdo a las consideraciones anteriores
# el resultado es asignar a todos los na dentro del dataframe el valor 0
eci[is.na(eci)] = 0
#Nota en trasaction_predweek
#Aparentemente esta variable debería ser booleana, de forma que 1 implica transacción, 0 no transacción.
#pero en nuestros datos esta variable aparece con los siguientes valores 0,1,2,3 y 4.
#Vamos a crear una nueva variable transaction: 1 cuando ha habido transacción (trasaction_predweek >= 1), 0 cuando no ha habido transacción (trasaction_predweek = 0) 
eci$transaction = eci$trasaction_predweek
eci$transaction[eci$transaction > 1] = 1
eci$transaction = as.factor(eci$transaction)
#attach(eci)
summary(eci)
plot(eci$count_trans)
plot(eci$count_trans,eci$trasaction_predweek)




#------------------------------------------------
#Análisis con arboles de decisión binarios
#------------------------------------------------
library(tree)

#vamos a crear un data frame que sin las variables trasaction_predweek ni totalTransactionReveneu

#eci_to_tree = subset(eci,select = -c(totalTransactionRevenue,trasaction_predweek,fullVisitorId,count_trans,items_cart,p_funnel))
eci_to_tree = subset(eci,select = -c(totalTransactionRevenue,trasaction_predweek,fullVisitorId,count_trans,items_cart,p_funnel))
#eci_to_tree = subset(eci,select = -c(totalTransactionRevenue,trasaction_predweek,fullVisitorId))
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

#some Exploratory analysis
#datos muy desbalanceados, vamos a quedarnos con el mismo número de datos de 0 que de 1.
id_transaction_0 = sample(which(eci_to_tree$transaction==0),sum(eci_to_tree$transaction == 1)+100)
#create new data frame with balanced data
balanced_eci = eci_to_tree[c(id_transaction_0,which(eci_to_tree$transaction==1)),]


summary(balanced_eci)



#exploratory analysis
#pairs(balanced_eci)
data_frame_for_corr = eci[c(id_transaction_0,which(eci_to_tree$transaction==1)),]
data_frame_for_corr$transaction = as.numeric(as.character(data_frame_for_corr$transaction))
data_frame_for_corr$newvisit = as.numeric(data_frame_for_corr$newvisit)
cor(data_frame_for_corr)
#pairs(data_frame_for_corr)

#look for features correlated with transaction
labs = names(data_frame_for_corr)
for (i in 1:length(data_frame_for_corr)){ 
  correlation_with_tr = cor(data_frame_for_corr[,i],data_frame_for_corr$transaction)
  correlation_with_tr_pred = cor(data_frame_for_corr[,i],data_frame_for_corr$trasaction_predweek)
  cat("Correlation ",labs[i],' vs. Transaction\n')
  cat("rho = ",correlation_with_tr,'\n')
  cat("Correlation ",labs[i],' vs. Transaction predweek\n')
  cat("rho = ",correlation_with_tr_pred,'\n')
  cat('------------------------\n')
}


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
    boxplot(balanced_eci[,i]~balanced_eci$transaction,xlab = 'transaction',ylab = labs[i])
  }
}


tree.eci = tree(balanced_eci$transaction~.,balanced_eci)
#tree.eci = tree(eci_to_tree$transaction~.,eci_to_tree)

summary(tree.eci)
plot(tree.eci)
text(tree.eci,pretty = 0)

#some pruning
set.seed(3)
cv.tree_eci = cv.tree(tree.eci, FUN=prune.misclass)

#vamos a probar con 5 que tiene la misma desvianza que con 8
prune.tree_eci = prune.misclass(tree.eci,best = 5)
plot(prune.tree_eci)
text(prune.tree_eci,pretty= 0 )
#----------------------------
# Logistic regression
#----------------------------


#En este punto vamos a pasar a realizar el análisis mediante regresión logística
library(glm)
eci.lgr(balanced_eci$)
