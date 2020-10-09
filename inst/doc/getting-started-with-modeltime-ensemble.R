## ---- include = FALSE---------------------------------------------------------
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

## ---- echo=F, out.width='100%', fig.align='center'----------------------------
knitr::include_graphics("stacking.jpg")

## ----setup--------------------------------------------------------------------
# Time Series ML
library(tidymodels)
library(modeltime)
library(modeltime.ensemble)

# Core
library(tidyverse)
library(timetk)

interactive <- FALSE

## -----------------------------------------------------------------------------
m750 %>%
    plot_time_series(date, value, .color_var = id, .interactive = interactive)

## -----------------------------------------------------------------------------
splits <- time_series_split(m750, assess = "2 years", cumulative = TRUE)

splits %>%
    tk_time_series_cv_plan() %>%
    plot_time_series_cv_plan(date, value, .interactive = interactive)

## -----------------------------------------------------------------------------
recipe_spec <- recipe(value ~ date, training(splits)) %>%
    step_timeseries_signature(date) %>%
    step_rm(matches("(.iso$)|(.xts$)")) %>%
    step_normalize(matches("(index.num$)|(_year$)")) %>%
    step_dummy(all_nominal()) %>%
    step_fourier(date, K = 1, period = 12)

recipe_spec %>% prep() %>% juice()

## -----------------------------------------------------------------------------
model_spec_arima <- arima_reg() %>%
    set_engine("auto_arima")

wflw_fit_arima <- workflow() %>%
    add_model(model_spec_arima) %>%
    add_recipe(recipe_spec %>% step_rm(all_predictors(), -date)) %>%
    fit(training(splits))

## -----------------------------------------------------------------------------
model_spec_prophet <- prophet_reg() %>%
    set_engine("prophet")

wflw_fit_prophet <- workflow() %>%
    add_model(model_spec_prophet) %>%
    add_recipe(recipe_spec %>% step_rm(all_predictors(), -date)) %>%
    fit(training(splits))

## -----------------------------------------------------------------------------
model_spec_glmnet <- linear_reg(
    mixture = 0.9,
    penalty = 4.36e-6
) %>%
    set_engine("glmnet")

wflw_fit_glmnet <- workflow() %>%
    add_model(model_spec_glmnet) %>%
    add_recipe(recipe_spec %>% step_rm(date)) %>%
    fit(training(splits))

## -----------------------------------------------------------------------------
m750_models <- modeltime_table(
    wflw_fit_arima,
    wflw_fit_prophet,
    wflw_fit_glmnet
)

m750_models

## -----------------------------------------------------------------------------
ensemble_fit <- m750_models %>%
    ensemble_average(type = "mean")

ensemble_fit

## -----------------------------------------------------------------------------
# Calibration
calibration_tbl <- modeltime_table(
    ensemble_fit
) %>%
    modeltime_calibrate(testing(m750_splits))

# Forecast vs Test Set
calibration_tbl %>%
    modeltime_forecast(
        new_data    = testing(m750_splits),
        actual_data = m750
    ) %>%
    plot_modeltime_forecast(.interactive = interactive)

## -----------------------------------------------------------------------------
refit_tbl <- calibration_tbl %>%
    modeltime_refit(m750)

refit_tbl %>%
    modeltime_forecast(
        h = "2 years",
        actual_data = m750
    ) %>%
    plot_modeltime_forecast(.interactive = interactive)

