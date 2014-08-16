---
layout: page
title: 
permalink: /julia_sparse/
---

## Julia Support: Can I use sparse input matrix?

Sure! We understand that many real-world applications involve sparse data
set, and therefore DimmWitted provides the support of using the native Julia
type `SparseMatrixCSC{T1,Int64}` directly. In this tutorial, we
walk through how to use this feature.
The code can be found
[here](https://github.com/zhangce/dw/blob/master/examples/julia_sparse.jl).

**Pre-requisites...** To understand this tutorial, we assume that you have
already familiar with the [Julia walkthorugh](/dw/julia/), and knows how to
write a logistic regression model with dense data.

### Prepare the Data Set

We still use the same synthetic data set that we created for
the [dense case](/dw/julia/) by using

{% highlight julia linenos%}
sparse_example=sparse(examples)
{% endhighlight %}

Here the variable `sparse_example` is of the type
`SparseMatrixCSC{Cdouble,Int64}`.

### Change the Function

Here is the last change you need to make to use sparse data!
The signature of the loss and gradient function needs to change
accordingly for sparse data:

{% highlight julia linenos%}
function loss(row, model::Array{Cdouble,1})
   ...
end
{% endhighlight %}

Here, the `row` object is not of the type `Array{Cdouble,1}`
any more, instead, it is an Array of

{% highlight julia linenos%}
immutable TMPTYPE
    idx::Clonglong
    data::Cdouble
end
{% endhighlight %}

where `idx` is the element index (for row access, it is the column id
for non-zero elements), and `data` is the actual data element.
It is not hard to see that `row` is a sparse representation
of a vector.

Given this difference, we can see that the following piece of code
implements a sparse version of the loss function.

{% highlight julia linenos%}
function loss(row, model::Array{Cdouble,1})
	const nfeat = length(model)
	const lastcol = nfeat + 1
	const nnz = length(row)
	label = 0.0
	if row[nnz].idx == lastcol
		label = row[nnz].data
	end
	d = 0.0
	for i = 1:nnz
		if row[i].idx != lastcol
			d = d + row[i].data*model[row[i].idx]
		end
	end
	return (-label * d + log(exp(d) + 1.0))
end
{% endhighlight %}

After this change, all other code is the same as the dense case.

### Can I Use Sparse Model?

As of DimmWitted v0.01, we do not support sparse model yet. Let us
know if you found that necessary in your application.
