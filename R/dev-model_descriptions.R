

#' @export
get_model_description.mdl_time_ensemble_avg <- function(object, indicate_training = FALSE, upper_case = TRUE) {

    type     <- object$parameters$type
    n_models <- object$n_models

    desc <- stringr::str_glue("Ensemble ({type}): {n_models} Models")

    if (indicate_training) {
        desc <- stringr::str_c(desc, " (Trained)")
    }

    if (upper_case) {
        desc <- toupper(desc)
    } else {
        desc <- tolower(desc)
    }

    return(desc)
}

#' @export
get_model_description.mdl_time_ensemble_wt <- function(object, indicate_training = FALSE, upper_case = TRUE) {

    n_models <- object$n_models

    desc <- stringr::str_glue("Ensemble (Weighted): {n_models} Models")

    if (indicate_training) {
        desc <- stringr::str_c(desc, " (Trained)")
    }

    if (upper_case) {
        desc <- toupper(desc)
    } else {
        desc <- tolower(desc)
    }

    return(desc)
}


#' @export
get_model_description.mdl_time_ensemble_model_spec <- function(object, indicate_training = FALSE, upper_case = TRUE) {

    n_models <- object$n_models
    desc     <- object$fit$fit %>% modeltime::get_model_description()

    desc <- stringr::str_glue("Ensemble ({desc} STACK): {n_models} Models")

    if (indicate_training) {
        desc <- stringr::str_c(desc, " (Trained)")
    }

    if (upper_case) {
        desc <- toupper(desc)
    } else {
        desc <- tolower(desc)
    }

    return(desc)
}

#' @export
get_model_description.recursive_ensemble <- function(object, indicate_training = FALSE, upper_case = TRUE) {

    class(object) <- class(object)[3:length(class(object))]

    desc <- get_model_description(object, indicate_training = FALSE, upper_case = TRUE)

    desc <- paste("RECURSIVE", desc)

    if (upper_case) {
        desc <- toupper(desc)
    } else {
        desc <- tolower(desc)
    }

    return(desc)

}
