context("TEST: ensemble_weighted()")

# TEST ENSEMBLE AVERAGE ----

# USED FOR TESTS ----
wflw_fit_arima <- workflow() %>%
    add_model(
        spec = arima_reg(seasonal_period = 12) %>% set_engine("auto_arima")
    ) %>%
    add_recipe(
        recipe = recipe(value ~ date, data = training(m750_splits))
    ) %>%
    fit(training(m750_splits))

wflw_fit_prophet <- workflow() %>%
    add_model(
        spec = prophet_reg() %>% set_engine("prophet")
    ) %>%
    add_recipe(
        recipe = recipe(value ~ date, data = training(m750_splits))
    ) %>%
    fit(training(m750_splits))

rec_glmnet <- recipe(value ~ date, data = training(m750_splits)) %>%
    step_timeseries_signature(date) %>%
    step_rm(matches("(iso$)|(xts$)|(am.pm)|(hour$)|(minute)|(second)")) %>%
    step_zv(all_predictors()) %>%
    step_normalize(all_numeric_predictors()) %>%
    step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
    step_rm(date)

# rec_glmnet %>% prep() %>% juice() %>% glimpse()


wflw_fit_glmnet <- workflow() %>%
    add_model(
        spec = linear_reg(penalty = 0.1) %>% set_engine("glmnet")
    ) %>%
    add_recipe(
        recipe = rec_glmnet
    ) %>%
    fit(training(m750_splits))

m750_models_2 <- modeltime_table(
    wflw_fit_arima,
    wflw_fit_prophet,
    wflw_fit_glmnet
)


# Median ----
test_that("ensemble_weighted()", {

    loadings <- c(3,2,1)

    ensemble_fit_wt <- m750_models_2 %>%
        ensemble_weighted(loadings = loadings)

    # Structure
    expect_s3_class(ensemble_fit_wt, "mdl_time_ensemble")
    expect_s3_class(ensemble_fit_wt, "mdl_time_ensemble_wt")

    expect_s3_class(ensemble_fit_wt$model_tbl, "mdl_time_tbl")
    expect_equal(ensemble_fit_wt$parameters$loadings, loadings)
    expect_true(ensemble_fit_wt$parameters$scale_loadings)

    expect_equal(ensemble_fit_wt$fit$loadings_tbl$.loadings, loadings / sum(loadings))

    expect_equal(ensemble_fit_wt$n_models, 3)
    expect_equal(ensemble_fit_wt$desc, "ENSEMBLE (WEIGHTED): 3 MODELS")

    # Print
    expect_equal(print(ensemble_fit_wt), ensemble_fit_wt)

    # Modeltime Table
    expect_equal(
        modeltime_table(ensemble_fit_wt) %>% pull(.model_desc),
        "ENSEMBLE (WEIGHTED): 3 MODELS"
    )

    # Forecast
    fcast <- modeltime_table(ensemble_fit_wt) %>%
        modeltime_forecast(testing(m750_splits))

    expect_equal(nrow(fcast), nrow(testing(m750_splits)))
    expect_equal(fcast$.index, testing(m750_splits)$date)

    # Calibration
    calibration_tbl <- ensemble_fit_wt %>%
        modeltime_calibrate(testing(m750_splits))

    expect_false(is.na(calibration_tbl$.type))

    # Accuracy
    accuracy_tbl <- calibration_tbl %>% modeltime_accuracy()

    expect_false(is.na(accuracy_tbl$mae))

    # Forecast
    forecast_tbl <- calibration_tbl %>%
        modeltime_forecast(
            new_data    = testing(m750_splits),
            actual_data = m750
        )

    n_actual <- nrow(m750)

    expect_equal(nrow(forecast_tbl), 24 + n_actual)
    expect_equal(ncol(forecast_tbl), 7)

    # Forecast - Test Keep New Data
    forecast_tbl <- calibration_tbl %>%
        modeltime_forecast(
            new_data    = testing(m750_splits),
            actual_data = m750,
            keep_data   = TRUE
        )

    expect_equal(nrow(forecast_tbl), 24 + n_actual)
    expect_equal(ncol(forecast_tbl), 10)

    # Refit
    refit_tbl <- calibration_tbl %>%
        modeltime_refit(m750)

    training_results_tbl <- refit_tbl %>%
        pluck(".model", 1, "model_tbl", ".model", 1, "fit", "fit", "fit", "data")

    expect_equal(nrow(training_results_tbl), nrow(m750))

})



# Checks/Errors ----
test_that("Checks/Errors: ensemble_weighted()", {

    # Object is Missing
    expect_error(ensemble_weighted())

    # Incorrect Object
    expect_error(ensemble_weighted(1))

    # No loadings
    expect_error(ensemble_weighted(m750_models_2))

    # Needs correct number of loadings
    expect_error({
        m750_models_2 %>%
            ensemble_weighted(loadings = 1)
    })

    # Needs more than one model
    expect_error({
        m750_models_2 %>%
            slice(1) %>%
            ensemble_weighted(loadings = 1:3)
    })

})



