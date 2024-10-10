import React from "react";
import { Redirect, Route, useHistory } from "react-router-dom";

interface IProtectedRouteProps {
  isAuthenticated: boolean;
  path: string;
  component: React.FC;
}

const ProtectedRoute = ({
  isAuthenticated,
  path,
  component,
}: IProtectedRouteProps) => {
  const history = useHistory();
  if (isAuthenticated) return <Route exact path={path} component={component} />;
  else {
    history.push("/sign-in");
    return <Redirect to="/sign-in" />;
  }

};

export default ProtectedRoute;
