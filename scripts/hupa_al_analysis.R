library(ggplot2)
library(dplyr)
library(brms)
library(ggeffects)
library(ggrepel)

asr_al_30 <- read.csv('../results/asr_ave_results_30.csv', header = T, sep = ',', row.names = NULL)
asr_al_30$Total <- asr_al_30$Size + asr_al_30$Select_size
asr_al_30$Method <- rep('AL', nrow(asr_al_30))
head(asr_al_30)

asr_al_60 <- read.csv('../results/asr_ave_results_60.csv', header = T, sep = ',', row.names = NULL)
asr_al_60$Total <- asr_al_60$Size + asr_al_60$Select_size
asr_al_60$Method <- rep('AL 60', nrow(asr_al_60))
head(asr_al_60)

asr_random <- read.csv('../results/asr_ave_results_random.txt', header = T, sep = ' ', row.names = NULL)
asr_random$Total <- asr_random$Size

asr_al <- rbind(asr_al_30, asr_al_60)
asr_al$Select_size <- as.numeric(asr_al$Select_size)
#asr_al$Size <- as.factor(asr_al$Size)

asr_al_select <- select(asr_al, c('Language', 'Task', 'Size', 'Actual_duration', 'Model', 'Metric', 'Value', 'Method', 'Total'))
asr_results <- rbind(asr_al_select, asr_random)

### Calculate the proportion of times random WER is larger than al WER
i = 1
c = 0
diff = 0
size_vector = rep(0, nrow(asr_random) / 2)
diff_vector = rep(0, nrow(asr_random) / 2)

for (size in as.vector(unique(asr_random$Total))){
  subset_data <- subset(asr_results, Metric == 'wer')
  al <- subset(subset_data, Method == 'AL 60')
  if (size %in% as.vector(al$Total)){
    al <- subset(al, Total == size)
    random <- subset(subset_data, Method == 'random' & Total == size)
    print(c(size, al$Value, random$Value))
    diff = diff + (random$Value - al$Value)
    size_vector[i] = size
    diff_vector[i] = random$Value - al$Value
    i = i + 1
    if (random$Value > al$Value){
      c = c + 1
    }
  }
}

c / (nrow(asr_random) / 2)
diff / (nrow(asr_random) / 2)

size_diff <- data.frame(size_vector)
names(size_diff) <- c('Total')
size_diff$Diff <- diff_vector


### How much lower is average WER 
al_30<-subset(asr_results, Method=='AL 30' & Metric=='wer')
random<-subset(asr_results, Method=='random' & Metric=='wer')
al_30 <- subset(al_30, Total %in% random$Total)
(mean(random$Value) - mean(al_30$Value)) / mean(random$Value)

al_60<-subset(asr_results, Method=='AL 60' & Metric=='wer')
random<-subset(asr_results, Method=='random' & Metric=='wer')
random <- subset(random, Total %in% al_60$Total)
(mean(random$Value) - mean(al_60$Value)) / mean(random$Value)


### Average WER plot
subset(asr_results, Metric == 'wer') %>%
  ggplot(aes(Total, Value, group = Method, color = Method, linetype=Method)) + 
  geom_point(aes(color = Method), alpha=.01) +
  geom_line() +
#  scale_color_manual(values = c("darkblue", "peru")) +
  #  scale_color_manual(values = wes_palette('Darjeeling2', n = 6)) + 
  #  scale_x_continuous(breaks=seq(0.5,5,1)) + 
  labs(x = "Training data duration (in minutes)") +
  labs(y = 'WER') +
  # geom_smooth(mapping = aes(Total, Value), alpha=0.45) +#, method = 'gam', linetype=Size) +
  #  facet_wrap(~ Metric, ncol = 8) +
  theme_classic() + 
  theme(legend.asrition="top")  + 
  theme(text = element_text(size=20, family="Times")) +
  ylim(0, 1) +
  scale_x_continuous(breaks = seq(30, 350, by = 20))

subset(asr_al, Metric == 'wer') %>%
  ggplot(aes(Total, Value, group = Size, color = Size, linetype=Size)) + 
  geom_point(aes(color = Size), alpha=.01) +
  geom_line() +
  scale_color_manual(values = c("darkblue", "peru")) +
  #  scale_color_manual(values = wes_palette('Darjeeling2', n = 6)) + 
  #  scale_x_continuous(breaks=seq(0.5,5,1)) + 
  labs(x = "Training data duration (in minutes)") +
  labs(y = 'WER') +
 # geom_smooth(mapping = aes(Total, Value), alpha=0.45) +#, method = 'gam', linetype=Size) +
  #  facet_wrap(~ Metric, ncol = 8) +
  theme_classic() + 
  theme(legend.asrition="top")  + 
  theme(text = element_text(size=20, family="Times")) +
  ylim(0, 1) +
  scale_x_continuous(breaks = seq(30, 350, by = 20))

## Smooth line
subset(asr_al, Metric == 'wer') %>%
  ggplot(aes(Total, Value, group = Size, color = Size, linetype=Size)) + 
  geom_point(aes(color = Size), alpha=.01) +
  #  geom_line(aes(linetype=Size)) +
  scale_color_manual(values = c("darkblue", "peru")) +
  #  scale_color_manual(values = wes_palette('Darjeeling2', n = 6)) + 
  #  scale_x_continuous(breaks=seq(0.5,5,1)) + 
  labs(x = "Training data duration (in minutes)") +
  labs(y = 'WER') +
  geom_smooth(mapping = aes(Total, Value), alpha=0.45) +#, method = 'gam', linetype=Size) +
  #  facet_wrap(~ Metric, ncol = 8) +
  theme_classic() + 
  theme(legend.asrition="top")  + 
  theme(text = element_text(size=20, family="Times")) +
  ylim(0, 1) +
  scale_x_continuous(breaks = seq(30, 350, by = 20))


asr_al$Size <- as.factor(asr_al$Size)
subset(asr_al, Metric == 'wer') %>%
  ggplot(aes(Total, Value, group = Size, color = Size)) + 
  geom_point(aes(color = Size), alpha=1) +
  #  geom_line(aes(linetype=Size)) +
  scale_color_manual(values = c("darkblue", "peru")) +
  # scale_color_brewer(palette = "Accent") +
  #  scale_color_manual(values = wes_palette('Darjeeling2', n = 6)) + 
  #  scale_x_continuous(breaks=seq(0.5,5,1)) + 
  labs(x = "Training data duration (in minutes)") +
  labs(y = 'WER') +
  # geom_smooth(mapping = aes(Total, Std), method = 'gam') +
  facet_wrap(~ Size, ncol = 1) +
  theme_classic() + 
  theme(legend.asrition="top")  + 
  theme(text = element_text(size=20, family="Times")) +
  ylim(0, 1) +
  scale_x_continuous(breaks = seq(30, 350, by = 20))

### WER std plot
subset(asr_al, Metric == 'wer') %>%
  ggplot(aes(Total, Std, group = Size, color = Size)) + 
  geom_point(aes(color = Size), alpha=1) +
  #  geom_line(aes(linetype=Size)) +
  scale_color_manual(values = c("darkblue", "peru")) +
  # scale_color_brewer(palette = "Accent") +
  #  scale_color_manual(values = wes_palette('Darjeeling2', n = 6)) + 
  #  scale_x_continuous(breaks=seq(0.5,5,1)) + 
  labs(x = "Training data duration (in minutes)") +
  labs(y = 'WER') +
 # geom_smooth(mapping = aes(Total, Std), method = 'gam') +
  facet_wrap(~ Size, ncol = 1) +
  theme_classic() + 
  theme(legend.asrition="top")  + 
  theme(text = element_text(size=20, family="Times")) +
  ylim(0, 0.15) +
  scale_x_continuous(breaks = seq(30, 350, by = 20))

### Average CER plot
subset(asr_al, Metric == 'cer') %>%
  ggplot(aes(Total, Value, group = Size, color = Size)) + 
  geom_point(aes(color = Size), alpha=.01) +
  geom_line(aes(linetype=Size)) +
  scale_color_manual(values = c("darkgreen",  "mediumpurple4")) +
  #  scale_color_manual(values = wes_palette('Darjeeling2', n = 6)) + 
  #  scale_x_continuous(breaks=seq(0.5,5,1)) + 
  labs(x = "Training data duration (in minutes)") +
  labs(y = 'CER') +
  #  geom_smooth(mapping = aes(Total, Value), method = 'gam') +
  #  facet_wrap(~ Metric, ncol = 8) +
  theme_classic() + 
  theme(legend.asrition="top")  + 
  theme(text = element_text(size=20, family="Times")) +
  ylim(0, 1) +
  scale_x_continuous(breaks = seq(30, 350, by = 20))


### WER drop
wer_al_30 <- subset(asr_al, Size == '30' & Metric == 'wer')
start_wer_30 <- subset(wer_al_30, Total == min(wer_al_30$Total))$Value
end_wer_30 <- subset(wer_al_30, Total == max(wer_al_30$Total))$Value
start_wer_30
end_wer_30

wer_al_60 <- subset(asr_al, Size == '60' & Metric == 'wer')
start_wer_60 <- subset(wer_al_60, Total == min(wer_al_60$Total))$Value
end_wer_60 <- subset(wer_al_60, Total == max(wer_al_60$Total))$Value
start_wer_60
end_wer_60

asr_al_30_1 <- read.csv('../results/asr_al_30_1.txt', header = T, sep = ' ', row.names = NULL)
asr_al_30_1$Total <- asr_al_30_1$Size + asr_al_30_1$Select_size
head(asr_al_30_1)

asr_al_30_2 <- read.csv('../results/asr_al_30_2.txt', header = T, sep = ' ', row.names = NULL)
asr_al_30_2$Total <- asr_al_30_2$Size + asr_al_30_2$Select_size
head(asr_al_30_2)

asr_al_30_3 <- read.csv('../results/asr_al_30_3.txt', header = T, sep = ' ', row.names = NULL)
asr_al_30_3$Total <- asr_al_30_3$Size + asr_al_30_3$Select_size
head(asr_al_30_3)

subset(asr_al_30_1, Metric == 'wer') %>%
  ggplot(aes(Total, Value, group = Size, color = Size)) + 
  geom_point(aes(color = Size), alpha=1) +
    geom_line() +
#  scale_color_manual(values = c("darkblue")) +
  # scale_color_brewer(palette = "Accent") +
  #  scale_color_manual(values = wes_palette('Darjeeling2', n = 6)) + 
  #  scale_x_continuous(breaks=seq(0.5,5,1)) + 
  labs(x = "Training data duration (in minutes)") +
  labs(y = 'WER') +
#  geom_smooth(mapping = aes(Total, Value), method = 'gam') +
  theme_classic() + 
    theme(text = element_text(size=20, family="Times")) +
  ylim(0, 1) +
  scale_x_continuous(breaks = seq(30, 350, by = 20))


### Fitting bounded exponential curve
wer_30_nonlinear <- brm(
  bf(Value  ~ 1 - (upperAsymptote - (upperAsymptote - 0) * exp(-Total * growthRate)),
     upperAsymptote ~ 1,
     growthRate ~ 1,
     nl = TRUE),
  data = wer_al_30,
  prior = c(
    prior(uniform(0, 1), nlpar = "upperAsymptote", lb = 0, ub = 1),
    prior(uniform(0, 10), nlpar = "growthRate", lb = 0, ub = 10)
  ),
  file = "wer_30_nonlinear",
  iter = 6000,
  warmup = 1000,
  chains = 4,
  cores = 4,
  thin = 4,
  control=list(adapt_delta=0.99))

wer_30_nonlinear_predictions <- as.data.frame(ggpredict(wer_30_nonlinear))
wer_30_nonlinear_predictions$Metric <- "wer"
wer_30_nonlinear_predictions$Size <- "30"

saveRDS(wer_30_nonlinear_predictions, "wer_30_nonlinear_predictions")

wer_30_nonlinear_predictions %>%
#  filter(valence=="negative", level=="sentence") %>%
  ggplot(aes(Total.x, Total.predicted, color=Size)) +
  geom_line(aes(color=Size)) +
  geom_label_repel(aes(label=Size), data=filter(asr_30_nonlinear_predictions, Total.x == 340 & Size == 30)) +
  geom_ribbon(aes(ymin = Total.conf.low, ymax = Total.conf.high, fill = Size), alpha=0.3, linetype=0) +
  xlab("Training data duration (in minutes)") + ylab("WER")+
  theme_linedraw() + theme(legend.position="none")

wer_30_nonlinear_predictions %>%
  #  filter(valence=="negative", level=="sentence") %>%
  ggplot(aes(Total.x, Total.predicted, color=Size)) +
  geom_errorbar(aes(color=Size, ymin=Total.conf.low, ymax = Total.conf.high)) +
  geom_label_repel(aes(label=Size), data=filter(wer_30_nonlinear_predictions, Total.x == 340 & Size == 30)) +
#  geom_ribbon(aes(ymin = Total.conf.low, ymax = Total.conf.high, fill = Size), alpha=0.3, linetype=0) +
  xlab("Training data duration (in minutes)") + ylab("WER")+
  theme_linedraw() + theme(legend.position="none") + 
  scale_x_continuous(breaks = seq(30, 350, by = 20))

wer_60_nonlinear <- brm(
  bf(Value  ~ 1 - (upperAsymptote - (upperAsymptote - 0) * exp(-Total * growthRate)),
     upperAsymptote ~ 1,
     growthRate ~ 1,
     nl = TRUE),
  data = wer_al_60,
  prior = c(
    prior(uniform(0, 1), nlpar = "upperAsymptote", lb = 0, ub = 1),
    prior(uniform(0, 10), nlpar = "growthRate", lb = 0, ub = 10)
  ),
  file = "wer_60_nonlinear",
  iter = 6000,
  warmup = 1000,
  chains = 4,
  cores = 4,
  thin = 4,
  control=list(adapt_delta=0.99))

wer_60_nonlinear_predictions <- as.data.frame(ggpredict(wer_60_nonlinear))
wer_60_nonlinear_predictions$Metric <- "wer"
wer_60_nonlinear_predictions$Size <- "60"

saveRDS(wer_60_nonlinear_predictions, "wer_60_nonlinear_predictions")

wer_60_nonlinear_predictions %>%
  #  filter(valence=="negative", level=="sentence") %>%
  ggplot(aes(Total.x, Total.predicted, color=Size)) +
  geom_line(aes(color=Size)) +
  geom_label_repel(aes(label=Size), data=filter(wer_60_nonlinear_predictions, Total.x == 340 & Size == 60)) +
  geom_ribbon(aes(ymin = Total.conf.low, ymax = Total.conf.high, fill = Size), alpha=0.3, linetype=0) +
  xlab("Training data duration (in minutes)") + ylab("WER")+
  theme_linedraw() + theme(legend.position="none")

random_nonlinear <- brm(
  bf(Value  ~ 1 - (upperAsymptote - (upperAsymptote - 0) * exp(-Total * growthRate)),
     upperAsymptote ~ 1,
     growthRate ~ 1,
     nl = TRUE),
  data = subset(asr_random, Metric == 'wer'),
  prior = c(
    prior(uniform(0, 1), nlpar = "upperAsymptote", lb = 0, ub = 1),
    prior(uniform(0, 10), nlpar = "growthRate", lb = 0, ub = 10)
  ),
  file = "random_nonlinear",
  iter = 6000,
  warmup = 1000,
  chains = 4,
  cores = 4,
  thin = 4,
  control=list(adapt_delta=0.99))

random_nonlinear_predictions <- as.data.frame(ggpredict(random_nonlinear))
random_nonlinear_predictions$Metric <- "wer"
random_nonlinear_predictions$Size <- "random"
random_nonlinear_predictions <- subset(random_nonlinear_predictions, Total.x %in% asr_random$Total)

saveRDS(random_nonlinear_predictions, "random_nonlinear_predictions")

random_nonlinear_predictions %>%
  #  filter(valence=="negative", level=="sentence") %>%
  ggplot(aes(Total.x, Total.predicted, color=Metric)) +
#  geom_line(aes(color=Metric)) +
#  geom_label_repel(aes(label=Size), data=random_nonlinear_predictions) +
  geom_errorbar(aes(ymin = Total.conf.low, ymax = Total.conf.high, color = Metric)) +
  xlab("Training data duration (in minutes)") + ylab("WER")+
  theme_linedraw() + theme(legend.position="none")


wer_nonlinear_predictions <-rbind(wer_30_nonlinear_predictions, random_nonlinear_predictions)

wer_nonlinear_predictions %>%
  #  filter(valence=="negative", level=="sentence") %>%
  ggplot(aes(Total.x, Total.predicted, color=Size)) +
  geom_label_repel(aes(label=Size), data=filter(wer_nonlinear_predictions, Total.x == 340)) +
  geom_errorbar(aes(ymin = Total.conf.low, ymax = Total.conf.high, color = Size)) +
  xlab("Training data duration (in minutes)") + ylab("WER")+
  theme_linedraw() + theme(legend.position="none") + 
  facet_wrap(~ Size, ncol = 1) 


### An alternative way of getting predictions and doing plotting
asr_30_nonlinear_predictions <- as.data.frame(predict(asr_30_nonlinear))
asr_30_nonlinear_predictions$Total <- asr_30_wer$Total
asr_30_nonlinear_predictions$Size <- rep('30', nrow(asr_30_nonlinear_predictions))

asr_30_nonlinear_predictions %>%
# filter(valence=="negative", level=="sentence") %>%
  ggplot(aes(Total, Estimate, color=Size)) +
#  geom_line(aes(color=Size)) +
  geom_errorbar(aes(color=Size, ymin=Q2.5, ymax = Q97.5)) +
#  geom_label_repel(aes(label=construction), data=filter(predictions, Age.x == 60, level=="sentence", valence=="negative")) +
#  geom_ribbon(aes(ymin = Q2.5, ymax = Q97.5, fill = Size), alpha=0.3, linetype=0) +
  xlab("Training data duration (in minutes)") + ylab("WER")+
  theme_linedraw() + theme(legend.position="none") +
  ylim(0, 1) +
  scale_x_continuous(breaks = seq(30, 350, by = 20))

