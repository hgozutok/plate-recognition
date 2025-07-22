import React from "react";
import { Link, useLocation } from "react-router-dom";

function Navbar() {
  const location = useLocation();
  const isActive = (path) => location.pathname === path;

  return (
    <nav className="bg-white shadow">
      <div className="container mx-auto px-4">
        <div className="flex justify-between h-16">
          <div className="flex">
            <Link to="/dashboard" className="flex items-center px-2 py-2">
              <span className="text-xl font-bold text-gray-800">
                Plate Recognition
              </span>
            </Link>
          </div>
          <div className="flex space-x-4">
            <NavLink to="/dashboard" isActive={isActive("/dashboard")}>
              Dashboard
            </NavLink>
            <NavLink to="/upload" isActive={isActive("/upload")}>
              Upload
            </NavLink>
            <NavLink to="/webcam" isActive={isActive("/webcam")}>
              Webcam
            </NavLink>
            <NavLink to="/history" isActive={isActive("/history")}>
              History
            </NavLink>
            <NavLink to="/login" isActive={isActive("/login")}>
              Login
            </NavLink>
          </div>
        </div>
      </div>
    </nav>
  );
}

function NavLink({ children, to, isActive }) {
  return (
    <Link
      to={to}
      className={`flex items-center px-3 py-2 rounded-md text-sm font-medium ${
        isActive ? "bg-gray-900 text-white" : "text-gray-600 hover:bg-gray-100"
      }`}
    >
      {children}
    </Link>
  );
}

export default Navbar;
