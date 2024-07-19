## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
    # collapse = TRUE,
    message = FALSE, 
    warning = FALSE,
    paged.print = FALSE,
    comment = "#>",
    fig.width = 8, 
    fig.height = 4.5,
    fig.align = 'center',
    out.width='95%'
)

## ----echo=F-------------------------------------------------------------------
knitr::include_graphics("panel-ensemble.png")

## -----------------------------------------------------------------------------
library(modeltime.ensemble)
library(modeltime)
library(tidymodels)
library(glmnet)
library(xgboost)
library(dplyr)
library(lubridate)
library(timetk)

## -----------------------------------------------------------------------------
FORECAST_HORIZON <- 24

m750_extended <- m750 %>%
    group_by(id) %>%
    future_frame(
        .length_out = FORECAST_HORIZON,
        .bind_data  = TRUE
    ) %>%
    ungroup()

## -----------------------------------------------------------------------------
lag_transformer <- function(data){
    data %>%
        tk_augment_lags(value, .lags = 1:FORECAST_HORIZON)
}

# Data Preparation
m750_lagged <- m750_extended %>% lag_transformer()
m750_lagged

## -----------------------------------------------------------------------------
train_data <- m750_lagged %>%
    filter(!is.na(value)) %>%
    tidyr::drop_na()

future_data <- m750_lagged %>%
    filter(is.na(value))

## ----eval=rlang::is_installed("earth")----------------------------------------
model_fit_lm <- linear_reg() %>%
    set_engine("lm") %>%
    fit(value ~ ., data = train_data %>% select(-id))

model_fit_mars <- mars("regression") %>%
    set_engine("earth", endspan = 24) %>%
    fit(value ~ ., data = train_data %>% select(-id))

## -----------------------------------------------------------------------------
recursive_ensemble <- modeltime_table(
    model_fit_lm,
    model_fit_mars
) %>%
    ensemble_average(type = "mean") %>%
    recursive(
        transform  = lag_transformer,
        train_tail = tail(train_data, FORECAST_HORIZON)
    )

recursive_ensemble

## -----------------------------------------------------------------------------
model_tbl <- modeltime_table(
    recursive_ensemble
)

model_tbl

## -----------------------------------------------------------------------------
model_tbl %>%
    modeltime_forecast(
        new_data    = future_data,
        actual_data = m750
    ) %>%
    plot_modeltime_forecast(
        .interactive        = FALSE,
        .conf_interval_show = FALSE,
    )

## -----------------------------------------------------------------------------
FORECAST_HORIZON <- 24

m4_extended <- m4_monthly %>%
    group_by(id) %>%
    future_frame(
        .length_out = FORECAST_HORIZON,
        .bind_data  = TRUE
    ) %>%
    ungroup()

## -----------------------------------------------------------------------------
lag_transformer_grouped <- function(data){
    data %>%
        group_by(id) %>%
        tk_augment_lags(value, .lags = 1:FORECAST_HORIZON) %>%
        ungroup()
}

## -----------------------------------------------------------------------------
m4_lags <- m4_extended %>%
    lag_transformer_grouped()

m4_lags

## -----------------------------------------------------------------------------
train_data <- m4_lags %>%
    tidyr::drop_na()

future_data <- m4_lags %>%
    filter(is.na(value))

## -----------------------------------------------------------------------------
model_fit_glmnet <- linear_reg(penalty = 1) %>%
    set_engine("glmnet") %>%
    fit(value ~ ., data = train_data)

model_fit_xgboost <- boost_tree("regression", learn_rate = 0.35) %>%
    set_engine("xgboost") %>%
    fit(value ~ ., data = train_data)

## -----------------------------------------------------------------------------
recursive_ensemble_panel <- modeltime_table(
    model_fit_glmnet,
    model_fit_xgboost
) %>%
    ensemble_weighted(loadings = c(4, 6)) %>%
    recursive(
        transform  = lag_transformer_grouped,
        train_tail = panel_tail(train_data, id, FORECAST_HORIZON),
        id         = "id"
    )

recursive_ensemble_panel

## -----------------------------------------------------------------------------
model_tbl <- modeltime_table(
    recursive_ensemble_panel
)

model_tbl

## -----------------------------------------------------------------------------
model_tbl %>%
    modeltime_forecast(
        new_data    = future_data,
        actual_data = m4_lags,
        keep_data   = TRUE
    ) %>%
    group_by(id) %>%
    plot_modeltime_forecast(
        .interactive        = FALSE,
        .conf_interval_show = FALSE,
        .facet_ncol         = 2
    )

