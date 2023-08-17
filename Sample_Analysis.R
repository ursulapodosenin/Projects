manager<-c(1,2,3,4,5)
date1<-c("10/21/14","9/5/14","7/8/14","10/15/14","5/3/14")
country<-c("UK","US", "UK", "UK", "US")
gender <- c("F","F","F","M","M")
age <- c(33,41,15,29,79)
q1 <- c(4,3,3,2,2)
q2 <- c(4,5,3,3,1)
q3 <- c(5,1,5,1,1)
q4 <- c(5,3,5,NA,3)
q5 <- c(5,4,2,NA,1)

#put data into dataframe
leadership <- data.frame(manager,date1,gender,country,age,q1,q2,q3,q4,q5, stringsAsFactors=FALSE)

#look at dataframe
leadership
head(leadership, n=2)
tail(leadership, n=1)
str(leadership) # structure: tells you what type each of your variables are

glimpse(leadership)

#indexing / selecting rows,columns
leadership[3,c(1,4)]
age_vector<-rbind(leadership$age, leadership$gender)


# Creating new variables/dataframe 
mydata<-data.frame(x1 = c(2, 2, 6, 4),
                   x2 = c(3, 4, 2, 8))


#mydata$avgx1x2<-rowMeans(mydata[,1:2])
dim(mydata) #find dimensions of dataframe
rowMeans(mydata) #average for each row
colMeans(mydata)#average for each column
mean(mydata$x1) #average all data in column 1
mydata$avgx1x2.1<-(mydata$x1+mydata$x2)/2 #long format of formula
mydata$avgx1x2.2<-rowMeans(mydata[,1:2]) #same thing with row means, but be careful, this will override the previous one!

dim(mydata)
mydata<-mydata[,-5] #removing extra 5th column we created in line 53

names(mydata) # gives you a list of your columns, which you can then use...
mydata[,"x1"] # ... to then call columns!

# method 1 -- $var_name or using numerical index (sum & mean & view data)
mydata[,1]
mydata[,"x1"]
mydata$x1

mydata$x1andx2<-mydata$x1+mydata$x2

# method 2 -- using attach/detach 
mydata2<-data.frame(x1 = c(2, 2, 6, 4),
                   x2 = c(3, 4, 2, 8))
dim(mydata2)

#attach()
attach(mydata2)
names(mydata2)
mydata2$x1andx2<-x1+x2
detach(mydata2)

#method 3 -- using transform
mydata3<-data.frame(x1 = c(2, 2, 6, 4),
                   x2 = c(3, 4, 2, 8))
mydata3 <- transform(mydata3,
                    x1andx2 = x1 + x2,
                    meanx = (x1 + x2)/2)
mydata3<-mydata3[,c('meanx', 'x1')]
names(mydata3)<-c('something_else', 'x1')
names(mydata3)
#mydata3$`something else`  #example of using name with spaces --> doable but not advised

## Method 4 - Using mutate from dplyr, as you did in your DataCamp assigment

#Syntax 1 - using forward pipes
mydata4 <- mydata %>% ## Load your data
  rowwise() %>% ## The default behavior is to do operations on columns. use this to override that
  mutate(x1andx2mean=mean(c(x1,x2))) # Make the new column using mutate and the mean function

## What are pipes? How do they work?
## Pipes take the output of whatever function you call and "pipe" or make it available to another function.
## Using these, you can string together commands without modifying the original object. This ends up being pretty useful,
## and can make your code a lot more efficient, especially in the context of dplyr.

# Syntax2 - without pipes

mydata5 <- mutate(rowwise(mydata), x1andx2mean=mean(c(x1,x2))) # Same thing, but no pipes


# Recoding variables: 
# method 1 - categorizing (65 elder, 35-65 Middle age, <35 young, 99 NA)
leadership<-leadership[,1:10]
leadership$age_cat[leadership$age > 65]<-"Elder"
#leadership$age_cat[which(leadership$age>65)]<-"Elder"
leadership$age_cat[leadership$age < 35]<-"Younger"
leadership$age_cat[leadership$age >35 & leadership$age < 65]<-"Middle-Age"
leadership$age_cat
# method 2 - using within()
leadership <- within(leadership,{
  agecat <- NA
  agecat[age > 75] <- "Elder"
  agecat[age >= 55 & age <= 75] <- "Middle Aged"
  agecat[age < 55] <- "Young" })


# Renaming variables
# method 1 - rename date to testDate
names(leadership)
which(names(leadership)=='date1')
names(leadership)[2]<-'NewDate' #renaming the second column in leadership to NewDate (was previously "date1")
names(leadership)
#names(leadership)[which(names(leadership)=='NewDate')]<-'date1'  #changing NewDate name back to date1


# method 2 - with plyr / dplyr
#install.packages('plyr')

# What is the difference between plyr and dplyr?
# Dplyr is best for data frames (or tibbles in the tidyverse), while plyr will work with many types of structures.

names(leadership)

library(plyr)
leadership2 <- plyr::rename(leadership,
                     c(manager="managerID", NewDate="testDate"))  # Use NewDate here (since we changed it from date1 in line 104)
head(leadership2) #notice that column 1 is now managerID and column 2 is now testDate

# Applying the is.na() function
y <- c(1, 2, 3, NA)
#with y
is.na(y)


#with leadership (q4)
is.na(leadership$q4)

# Recode 99 to missing for the variable age
leadership[is.na(leadership)]=999999
leadership

leadership[leadership==999999]=NaN
leadership

# Excluding missing values from analyses
x <- c(1, 2, NA, 3)

#manually add each (using indices)
x[1]+x[2]+x[3]+x[4] #Answer is NA bc there is an NA is in vector
x[1]+x[2]+x[4] #excluding x[3] since this is an NA value 

#add using sum()
sum(x) #if NA's are in vector, answer will be NA without na.rm=T argument (see below)
#add using sum() with na.rm argument
sum(x, na.rm = T)


# Using na.omit() to delete incomplete observations
x_noNA<-na.omit(x)
x_noNA #removed NA
# Converting character values to dates
mydates <- as.Date(c("2007-06-22", "2004-02-13"))
mydates 
str(mydates)
str(leadership$NewDate)

strDates <- c("01/05/1965", "08/16/1975") #US Jan 5, 1965; Aug 16, 1975
strDates2 <- c("01/05/1965", "08/12/1975") #EU --> May 1, 1965; Dec 8 1975

dates <- as.Date(strDates, "%m/%d/%Y")
dates2<- as.Date(strDates2, "%d/%m/%Y")
dates
#format(dates2, format="%m %d %y")

myformat <- "%m/%d/%y"
leadership$NewDate2 <- as.Date(leadership$NewDate, myformat)
names(leadership)
leadership$NewDate2

# Woring with formats (already a date value, just changing format of date)
today <- Sys.Date()
format(today, format="%B %d %Y")
format(today, format="%A")


# Calculations with dates
startdate <- as.Date("2004-02-13")
enddate   <- as.Date("2009-06-22")
enddate - startdate


# Date functions and formatted printing
today <- Sys.Date()
dob <- as.Date("1956-10-12")
difftime(today, dob, units="auto")


# Converting from one data type to another
a <- c(1,2,3)
#testing boolean statements
#is.numeric()
is.numeric(a)
#is.vector()
is.vector(a)
#try to add/subtract
a-100
#convert a to character (as.char)
a <- as.character(a)
a
#test numeric, vector, character
is.vector(a)
is.numeric(a)
is.character(a)

#try to add/subtract
a-100 #error because you can't perform mathematical operations on characters!
### sorting 
x <- sample(1:10, 5)

#order() (increasing and decreasing)
order(x)

# Sorting a dataset
newdata <- leadership[order(leadership$age),]
newdata

#order multiple variables within a dataframe
attach(leadership)
newdata <- leadership[order(gender, age),]
detach(leadership)

attach(leadership)
newdata <-leadership[order(gender, -age),]
detach(leadership)


# Selecting variables (I may sometimes refer to as "indexing" when using numerical column/row #'s)
newdata <- leadership[, c(5:9)]
#names of columns
myvars <- c("q1", "q2", "q3", "q4", "q5")


# Dropping variables
myvars <- names(leadership) %in% c("q3", "q4") 
leadership[!myvars] #! means "not"

myvars <- names(leadership) %in% c("q3", "q4") 

names(leadership) 
names(leadership) %in% c("q3", "q4") 


leadership[!myvars]



# Selecting observations (rows)
#grab first 3 rows
newdata <- leadership[1:3,]
newdata
#grab all rows of males over 30
newdata <- leadership[leadership$gender=="M" & leadership$age > 30,]
newdata

# Selecting observations based on dates
startdate <- as.Date("2009-01-01")
enddate <- as.Date("2009-10-31")
newdata <- leadership[which(leadership$date >= startdate &
                              leadership$date <= enddate),]
newdata


# Using the subset() function
newdata <- subset(leadership, age >= 35 | age < 24,
                  select=c(q1, q2, q3, q4))

#use subset() to grab males over 25
newdata <- subset(leadership, gender=="M" & age > 25,
                  select=gender:q4)




