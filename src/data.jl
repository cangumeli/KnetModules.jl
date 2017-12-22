using Knet:MB

import Knet.nll
import Knet.accuracy

"""
`nll(model::KnetModule, data::Knet.MB; o...)`
wrapper over `Knet.nll`.
"""
nll(model::KnetModule, data::MB; o...) =
    nll(model, data, (m, x)->(@run m(x)); o...)


"""
`accuracy(model::KnetModule, data::Knet.MB; o...)` 
wrapper over `Knet.accuracy`.
"""
accuracy(model::KnetModule, data::MB; o...) =
    accuracy(model, data, (m, x)->(@run m(x)); o...)

