---
layout: page
title: 
permalink: /julia_immutable/
---

## Julia Support: Can I use non-primative data type, e.g., structure, in my data?

Yes, you can. In this tutorial, we will walkthrough an example
of using non-primative type, both for data and model. The takeaway
is that as long as your data type is _immutable_, you can use it in 
DimmWitted in the same way as those primitive types. The code
used in this tutorial can be found [here](https://github.com/zhangce/dw/blob/master/examples/julia_composite.jl).

**Pre-requisites...** To understand this tutorial, we assume that you have
already familiar with the [Julia walkthorugh](/dw/julia/), and knows how to
write a logistic regression model where both data and model are of the
type `Cdouble`.

### Defining Data Type

The very first step of using non-primative type is to define the type that you want to use. DimmWitted is able to use any primative type that you defined.
For example, 

{% highlight julia linenos%}
immutable DoublePair
	d1::Cdouble
	d2::Cdouble
end
{% endhighlight %}

This piece of code creates a type called `DoublePair` that consists of
two `Cdouble` pairs. After we define this type, we can generate the data
in a similar way as [Julia walkthorugh](/dw/julia/). For example,
one application we can try is to train two logistic regression 
with complementary label (Line 9 and Line 11) in a single pass
over the data:

{% highlight julia linenos%}
nexp = 100000
nfeat = 1024
examples = Array(DoublePair, nexp, nfeat+1)
for row = 1:nexp
	for col = 1:nfeat
		examples[row, col] = DoublePair(1.0,1.0)
	end
	if rand() > 0.8
		examples[row, nfeat+1] = DoublePair(1.0,0.0)
	else
		examples[row, nfeat+1] = DoublePair(0.0,1.0)
	end
end
model = DoublePair[DoublePair(0.0,0.0) for i = 1:nfeat]
{% endhighlight %}

### Defining Functions

Given the new data type, to write the function (e.g., loss),
we only need to change the signature accordingly, for example

{% highlight julia linenos%}
function loss(row::Array{DoublePair,1}, model::Array{DoublePair,1})
	const label1 = row[length(row)].d1
	const label2 = row[length(row)].d2
	const nfeat = length(model)
	d1 = 0.0
	d2 = 0.0
	for i = 1:nfeat
		d1 = d1 + row[i].d1*model[i].d1
		d2 = d2 + row[i].d2*model[i].d2
	end
	v1 = (-label1 * d1 + log(exp(d1) + 1.0))
	v2 = (-label2 * d2 + log(exp(d2) + 1.0))
	return v1 + v2
end
{% endhighlight %}

Compared with [Julia walkthorugh](/dw/julia/), 
this piece of code calculates two losses and return the
sum of it. You can write the gradient function
in a way similar to the loss function.

### Getting the result!

After you define the function, you do not need to change anything
else to run DimmWitted! To validate our result,
we can check the result of the following piece of code:

{% highlight julia linenos%}
sum1 = 0.0
sum2 = 0.0
for i = 1:length(model)
	sum1 = sum1 + model[i].d1
	sum2 = sum2 + model[i].d2
end
println("SUM OF MODEL1: ", sum1)
println("SUM OF MODEL2: ", sum2)
{% endhighlight %}

The result should be

    SUM OF MODEL1: -1.2475164859764791
    SUM OF MODEL2: 1.2474677445561093

Which is consistent with the synthetic data that we just generated.

### Possible Pitfalls

There are couple things you need to keep in mind:

  - In DimmWitted v0.01, you must use `immutable`. For now, you
  **cannot** use `type`, `tuple`, or other ways of constructing composite types. 
  - Because you are using `immutable`, you cannot write things like

{% highlight julia linenos%}
model[i].d1 = 5.0
{% endhighlight %}

instead you need to write 

{% highlight julia linenos%}
model[i] = DoublePair(5, model[i].d2)
{% endhighlight %}




