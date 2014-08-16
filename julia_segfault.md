---
layout: page
title: 
permalink: /julia_segfault/
---

## Julia Support: I am getting a Segmentation Fault, or an ERROR saying my function ''protentially is not thread-safe''! What should I do?

DimmWitted calles Julia functions you wrote from multiple threads to increase
the performance. Because the current Julia engine is single-threaded,
not all Julia functions can be used successfully with DimmWitted. Therefore,
in DimmWitted, we provide

  1. A built-in simple sanity-checker to make our best effort guess in deciding
    whether your function can be used at register time. Of course, you can suppress the decision this sanity checker made as shown in this tutorial to use
    it at your own risk.
  2. We document in this page some best effort guidelines in designing Julia   
    functions that can be used in DimmWitted.
  3. We provide a debugging mode in DimmWitted to call your function from a 
    single thread to help you diagnose the problem.


**Pre-requisites...** To understand this tutorial, we assume that you have
already familiar with the [Julia walkthorugh](/dw/julia/).

### Built-in Sanity Checker

By default, everytime you call a function like `register_row`, DimmWitted
runs a sanity-checker to make a guess of whether this function can be
successfully used in DimmWitted. For example, if you register a function
like 

{% highlight julia linenos%}
function loss(row::Array{Cdouble,1}, model::Array{Cdouble,1})
        const label = row[length(row)]
        const nfeat = length(model)
        d = dot(row[1:nfeat], model)
        return (-label * d + log(exp(d) + 1.0))
end
{% endhighlight %}

You will see DimmWitted tells you

    ERROR: Your function contains LLVM LR `alloc` or `call` other julia 
    functions. We cannot register this function because it protentially 
    is not thread-safe. Use register_row(_dw,loss,true) to register this 
    function AT YOUR OWN RISK!

This means that DimmWitted's sanity checker thinks the function
you try to register might not be able to be successfully executed.
Of course, you can suppress the decision of this sanity checker
by putting `true` to the third argument of register_row. In this case,
DimmWitted will not complain and just use your function. In our example,
you can use 

{% highlight julia linenos%}
handle_loss = DimmWitted.register_row(dw, loss, true)
{% endhighlight %}

Unfortunately, in our example, the sanity checker is correct, and you will
see

    signal (11): Segmentation fault
    Segmentation fault (core dumped)

### How to Write DimmWitted-friendly Julia Code?

It is actually not very hard to write a DimmWitted-friendly
Julia code, and the core principle is to avoid memory allocation
or calling other functions that are not thread-safe, e.g., `dot`.

#### Writing Type-Stable Code

One can find a good tutorial [here](http://www.johnmyleswhite.com/notebook/2013/12/06/writing-type-stable-code-in-julia/) about how to
write type-stable Julia code. Let's see one example

{% highlight julia linenos%}
function loss(row::Array{Cdouble,1}, model::Array{Cdouble,1})
        const label = row[length(row)]
        const nfeat = length(model)
        d = 0
        for i = 1:nfeat
                d = d + row[i]*model[i]
        end
        return (-label * d + log(exp(d) + 1.0))
end
{% endhighlight %}

Can this function be used with DimmWitted? If you run this function,
our sanity checker will fails. If we check the LLVM LR using `code_llvm`,
you can see the following line that causes the problem:

    %2 = alloca [5 x %jl_value_t*], align 8

How should we revise our code to avoid this problem? We observe that
the problem is acutally caused by Line 4

{% highlight julia linenos%}
d = 0
{% endhighlight %}

In this line, `d` is of the type `Int64`, and at Line 6, `d`'s type changed
and becomes `Cdouble`. This type-change causes allocation of
the memory. Instead, if we replace Line 4 with

{% highlight julia linenos%}
d = 0.0
{% endhighlight %}

or 

{% highlight julia linenos%}
d = 0::Cdouble
{% endhighlight %}

We get a function that the sanity checker is happy with.

This trick is not mysterious, and one can consult
[this blog page](http://www.johnmyleswhite.com/notebook/2013/12/06/writing-type-stable-code-in-julia/) about how to write ''Type-Stable Code'' in Julia.

### Debug Mode

Our previous discussion builds upon the hypothesis that
it is multi-threading that causes the problem. Therefore,
to help you diagnose your problem, this mode is called 
`DimmWitted.MR_SINGLETHREAD_DEBUG`, and you can use it in
the way like

{% highlight julia linenos%}
dw = DimmWitted.open(examples, model,
                DimmWitted.MR_SINGLETHREAD_DEBUG,
                DimmWitted.DR_SHARDING,
                DimmWitted.AC_ROW)
{% endhighlight %}

If you use this, we can see that the version with `dot` can actually
run even though the sanity checker still complains.








