package halil.todolist.security.filter;

import com.auth0.jwt.JWT;
import com.auth0.jwt.algorithms.Algorithm;
import halil.todolist.security.SecurityConstants;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.filter.OncePerRequestFilter;

import javax.servlet.FilterChain;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.Arrays;

public class JWTAuthorizationFilter extends OncePerRequestFilter {

    // Authorization: Bearer JWT
    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain) throws ServletException, IOException {
        String header = request.getHeader("Authorization");    // Bearer JWT

        // 이 로직이 없다면 방금 회원가입 한 사람은 인증헤더가 null 이기 때문에 아래 로직 수행 X
        // 사용자가 방금 가입한 경우에는 인증 헤더를 전달할 필요가 없기 때문에 if문 검증
        if (!hasHeader(request)) {
            filterChain.doFilter(request, response);
            return;
        }

        String token = header.replace(SecurityConstants.BEARER, "");

        // JWT.require : 추출 기능
        // .build() ~ 로 디코딩된 JWT 를 반환
        String member = JWT.require(Algorithm.HMAC256(SecurityConstants.SECRET_LEY))
                .build()
                .verify(token)
                .getSubject();

        // 디코딩된 member는 principal, 암호는 credentials 에는 넣지 않는다(민감한 데이터가 포함되지 않음)
        Authentication authentication = new UsernamePasswordAuthenticationToken(member, null, Arrays.asList());

        // SCH에 인증 개체를 설정
        SecurityContextHolder.getContext().setAuthentication(authentication);
        filterChain.doFilter(request, response);
    }

    private Boolean hasHeader(HttpServletRequest request) {
        return request.getHeader("Authorization") != null && request
                .getHeader("Authorization")
                .startsWith(SecurityConstants.BEARER);
    }

//    public String getMemberEmail(String token) {
//        return
//    }
}
