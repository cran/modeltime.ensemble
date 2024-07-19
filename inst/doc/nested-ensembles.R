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

## -----------------------------------------------------------------------------
library(tidymodels)
library(modeltime)
library(modeltime.ensemble)
library(dplyr)
library(timetk)
library(gt)

## -----------------------------------------------------------------------------
data_tbl <- walmart_sales_weekly %>%
    select(id, date = Date, value = Weekly_Sales) %>%
    filter(id %in% c("1_1", "1_3"))

data_tbl

## -----------------------------------------------------------------------------
data_tbl %>%
  group_by(id) %>%
  plot_time_series(date, value, .facet_ncol = 1, .interactive = FALSE)

## -----------------------------------------------------------------------------
nested_data_tbl <- data_tbl %>%
    # Step 1: Extend
    extend_timeseries(
        .id_var        = id,
        .date_var      = date,
        .length_future = 52
    ) %>%
    # Step 2: Nest
    nest_timeseries(
        .id_var        = id,
        .length_future = 52,
        .length_actual = 52*2
    ) %>%
    # Step 3: Split Train/Test
    split_nested_timeseries(
        .length_test = 52
    )

nested_data_tbl

## -----------------------------------------------------------------------------
rec_prophet <- recipe(value ~ date, extract_nested_train_split(nested_data_tbl)) 

wflw_prophet <- workflow() %>%
    add_model(
        prophet_reg("regression", seasonality_yearly = TRUE) %>% 
            set_engine("prophet")
    ) %>%
    add_recipe(rec_prophet)

## -----------------------------------------------------------------------------
rec_xgb <- recipe(value ~ ., extract_nested_train_split(nested_data_tbl)) %>%
    step_timeseries_signature(date) %>%
    step_rm(date) %>%
    step_zv(all_predictors()) %>%
    step_dummy(all_nominal_predictors(), one_hot = TRUE)

wflw_xgb <- workflow() %>%
    add_model(boost_tree("regression") %>% set_engine("xgboost")) %>%
    add_recipe(rec_xgb)

## ----message=TRUE-------------------------------------------------------------
nested_modeltime_tbl <- modeltime_nested_fit(
  # Nested data 
  nested_data = nested_data_tbl,
  
  # Add workflows
  wflw_prophet,
  wflw_xgb
)

nested_modeltime_tbl

## -----------------------------------------------------------------------------
tab_style_by_group <- function(object, ..., style) {
  
  subset_log <- object[["_boxhead"]][["type"]]=="row_group"
  grp_col    <- object[["_boxhead"]][["var"]][subset_log] %>% rlang::sym()
  
  object %>%
    tab_style(
      style = style,
      locations = cells_body(
        rows = .[["_data"]] %>%
          tibble::rowid_to_column("rowid") %>%
          group_by(!!grp_col) %>%
          filter(...) %>%
          ungroup() %>%
          pull(rowid)
      )
    )
}

## -----------------------------------------------------------------------------
nested_modeltime_tbl %>% 
  extract_nested_test_accuracy() %>%
  group_by(id) %>%
  table_modeltime_accuracy(.interactive = FALSE) %>%
  tab_style_by_group(
    rmse == min(rmse),
    style = cell_fill(color = "lightblue")
  )
  

## -----------------------------------------------------------------------------
nested_ensemble_1_tbl <- nested_modeltime_tbl %>%
    ensemble_nested_average(
        type           = "mean", 
        keep_submodels = TRUE
    )

nested_ensemble_1_tbl

## -----------------------------------------------------------------------------
nested_ensemble_1_tbl %>% 
  extract_nested_test_accuracy() %>%
  group_by(id) %>%
  table_modeltime_accuracy(.interactive = FALSE) %>%
  tab_style_by_group(
    rmse == min(rmse),
    style = cell_fill(color = "lightblue")
  )

## ----message=TRUE-------------------------------------------------------------
nested_ensemble_2_tbl <- nested_ensemble_1_tbl %>%
    ensemble_nested_weighted(
        loadings        = c(2,1),  
        metric          = "rmse",
        model_ids       = c(1,2), 
        control         = control_nested_fit(allow_par = FALSE, verbose = TRUE)
    ) 

nested_ensemble_2_tbl

## -----------------------------------------------------------------------------
nested_ensemble_2_tbl %>% 
  extract_nested_test_accuracy() %>%
  group_by(id) %>%
  table_modeltime_accuracy(.interactive = FALSE) %>%
  tab_style_by_group(
    rmse == min(rmse),
    style = cell_fill(color = "lightblue")
  )

## -----------------------------------------------------------------------------
best_nested_modeltime_tbl <- nested_ensemble_2_tbl %>%
    modeltime_nested_select_best(
      metric                = "rmse", 
      minimize              = TRUE, 
      filter_test_forecasts = TRUE
    )

## -----------------------------------------------------------------------------
best_nested_modeltime_tbl %>%
  extract_nested_best_model_report() %>%
  table_modeltime_accuracy(.interactive = FALSE)

## -----------------------------------------------------------------------------
best_nested_modeltime_tbl %>%
  extract_nested_test_forecast() %>%
  group_by(id) %>%
  plot_modeltime_forecast(
    .facet_ncol  = 1,
    .interactive = FALSE
  )

## ----message=TRUE-------------------------------------------------------------
nested_modeltime_refit_tbl <- best_nested_modeltime_tbl %>%
    modeltime_nested_refit(
        control = control_nested_refit(verbose = TRUE)
    )

## ----message = TRUE-----------------------------------------------------------
nested_modeltime_refit_tbl

## -----------------------------------------------------------------------------
nested_modeltime_refit_tbl %>%
  extract_nested_future_forecast() %>%
  group_by(id) %>%
  plot_modeltime_forecast(
    .interactive = FALSE,
    .facet_ncol  = 2
  )

