package halil.todolist.security.filter.testFilter;

import javax.servlet.*;
import javax.servlet.http.HttpServletRequest;
import java.io.IOException;

public class TfilterOne implements Filter {

    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain) throws IOException, ServletException {
        System.out.println(((HttpServletRequest) request).getRequestURI());
        System.out.println("HEY WI ARE IN FILTERONE");
        chain.doFilter(request, response);
    }
}
