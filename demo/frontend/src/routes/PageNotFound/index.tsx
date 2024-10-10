import React from "react";
import pageNotFound from "assets/svgs/notFound.svg";
import "./index.css";

const PageNotFound = ()=>{
    return(
        <div className="page-not-found">
            <img src={pageNotFound} alt="Page Not Found" />
            <h1>Page not Found</h1>
        </div>
    )
}

export default PageNotFound;