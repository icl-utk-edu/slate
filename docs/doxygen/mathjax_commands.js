// see https://stackoverflow.com/questions/40270302/latex-newcommand-in-doxgygen-html-output
// see http://docs.mathjax.org/en/latest/tex.html#defining-tex-macros
MathJax.Hub.Config({
    TeX: {
        Macros: {
            abs: ["\\left| #1 \\right|", 1],
            norm: ["\\left\\lVert #1 \\right\\rVert", 1],
        }
    }
});
