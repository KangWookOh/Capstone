package halil.todolist.security.filter.testFilter;

import com.fasterxml.jackson.databind.ObjectMapper;
import halil.todolist.domain.member.entity.Member;

import javax.servlet.*;
import javax.servlet.http.HttpServletRequest;
import java.io.IOException;

public class TfilterTwo implements Filter {
    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain) throws IOException, ServletException {
        Member member = new ObjectMapper().readValue(((HttpServletRequest)request).getInputStream(), Member.class);
        System.out.println(member.getEmail());
        System.out.println(member.getPassword());
    }
}
